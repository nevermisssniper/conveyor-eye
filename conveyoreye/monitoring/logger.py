"""Async inference logger — persist every prediction without blocking serving.

WHY async + batched writes?
----------------------------
The serving hot path must not pay disk latency. If each ``/infer`` call wrote a
row synchronously, p99 latency would be hostage to SQLite's fsync. Instead the
API hands finished results to an ``asyncio.Queue`` (a near-instant in-memory put)
and a single background ``_writer_loop`` drains the queue and writes in batches
inside one transaction. This decouples request latency from write throughput and
turns thousands of tiny INSERTs into a handful of batched commits.

WHY two tables?
---------------
``frames`` is one row per inference (latency, frame-level max confidence, image
stats) — this is the grain drift queries run over. ``detections`` is one row per
box (class, confidence) — the grain the confidence-distribution drift detector
needs. Splitting them keeps each query scanning only what it needs, and the
indexes are built specifically for the *time-window* access pattern drift uses
("give me every detection confidence in the last hour").

Scalability path
----------------
  v1 (here): single-file SQLite, one writer task, batched transactions.
  v2: WAL mode + a partitioned/rotated DB per day; this interface is unchanged.
  v3: swap the SQLite backend for a columnar store / OLAP sink (DuckDB, ClickHouse)
      behind the same ``log()`` / ``query_*`` methods — the call sites don't move.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id      TEXT,
    ts            REAL NOT NULL,          -- unix seconds, epoch float
    n_detections  INTEGER NOT NULL,
    max_conf      REAL NOT NULL,          -- frame-level uncertainty signal
    latency_ms    REAL NOT NULL,
    brightness    REAL,                   -- image-stat proxies for drift
    contrast      REAL,
    edge_density  REAL
);
CREATE TABLE IF NOT EXISTS detections (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_pk  INTEGER NOT NULL,           -- FK -> frames.id
    ts        REAL NOT NULL,              -- denormalized for index-only time scans
    class_id  INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    FOREIGN KEY (frame_pk) REFERENCES frames(id)
);
-- Indexes built for the drift access pattern: time-windowed scans, optionally
-- per class. Without these, every drift check is a full table scan.
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts);
CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);
CREATE INDEX IF NOT EXISTS idx_det_class_ts ON detections(class_id, ts);
"""


@dataclass
class FrameRecord:
    """A normalized, log-ready view of one DetectionResult.

    Built by ``InferenceLogger.log`` so the queue carries plain data, not live
    model objects — the writer task never touches detector internals.
    """

    frame_id: str | None
    ts: float
    n_detections: int
    max_conf: float
    latency_ms: float
    brightness: float | None
    contrast: float | None
    edge_density: float | None
    detections: list[tuple[int, str, float, float, float, float, float]] = field(default_factory=list)
    # each detection tuple: (class_id, class_name, confidence, x1, y1, x2, y2)


class InferenceLogger:
    """Buffered async writer for inference results.

    Lifecycle is explicit: ``await start()`` spins up the writer task and creates
    the schema; ``await stop()`` flushes the queue and closes the DB. The FastAPI
    lifespan owns this lifecycle (see serving/api.py).
    """

    def __init__(
        self,
        db_path: str | Path,
        batch_size: int = 64,
        flush_interval_s: float = 2.0,
        max_queue: int = 10_000,
    ) -> None:
        self.db_path = str(db_path)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._queue: asyncio.Queue[FrameRecord] = asyncio.Queue(maxsize=max_queue)
        self._db: aiosqlite.Connection | None = None
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Open the DB, create schema, and launch the background writer."""
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")  # concurrent reads while writing
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        self._stopping.clear()
        self._task = asyncio.create_task(self._writer_loop(), name="inference-log-writer")

    async def stop(self) -> None:
        """Signal shutdown, drain remaining records, close the DB cleanly."""
        self._stopping.set()
        if self._task is not None:
            await self._task
            self._task = None
        if self._db is not None:
            await self._db.close()
            self._db = None

    # --------------------------------------------------------------- ingest

    async def log(self, result, image_stats: dict | None = None) -> None:
        """Enqueue a DetectionResult for persistence. Non-blocking on the hot path.

        Drops the record (rather than blocking serving) if the queue is full —
        losing a log line under extreme load is preferable to adding latency to a
        live inference. A dropped-count metric would live here in v2.
        """
        rec = self._to_record(result, image_stats)
        try:
            self._queue.put_nowait(rec)
        except asyncio.QueueFull:
            pass  # TODO(v2): increment a dropped-records counter for observability

    def _to_record(self, result, image_stats: dict | None) -> FrameRecord:
        stats = image_stats or {}
        dets = [
            (d.class_id, d.class_name, d.confidence, *d.xyxy)
            for d in result.detections
        ]
        return FrameRecord(
            frame_id=result.frame_id,
            ts=time.time(),
            n_detections=len(result.detections),
            max_conf=result.max_confidence,
            latency_ms=result.latency_ms,
            brightness=stats.get("brightness"),
            contrast=stats.get("contrast"),
            edge_density=stats.get("edge_density"),
            detections=dets,
        )

    # --------------------------------------------------------- writer loop

    async def _writer_loop(self) -> None:
        """Drain the queue in batches and commit; flush on size OR time.

        Two triggers matter: ``batch_size`` bounds memory and commit size under
        load; ``flush_interval_s`` bounds *staleness* under light load so a slow
        trickle of frames still lands within a couple seconds (drift queries want
        fresh data). On shutdown we drain whatever remains before returning.
        """
        assert self._db is not None
        buffer: list[FrameRecord] = []
        last_flush = time.monotonic()

        while not (self._stopping.is_set() and self._queue.empty()):
            timeout = max(0.05, self.flush_interval_s - (time.monotonic() - last_flush))
            try:
                rec = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                buffer.append(rec)
            except asyncio.TimeoutError:
                pass  # time-based flush below

            should_flush = (
                len(buffer) >= self.batch_size
                or (buffer and (time.monotonic() - last_flush) >= self.flush_interval_s)
                or (self._stopping.is_set() and buffer)
            )
            if should_flush:
                await self._flush(buffer)
                buffer.clear()
                last_flush = time.monotonic()

    async def _flush(self, buffer: list[FrameRecord]) -> None:
        """Write a batch of frames + their detections in a single transaction."""
        assert self._db is not None
        if not buffer:
            return
        async with self._db.execute("BEGIN"):
            pass
        for rec in buffer:
            cur = await self._db.execute(
                "INSERT INTO frames (frame_id, ts, n_detections, max_conf, latency_ms,"
                " brightness, contrast, edge_density) VALUES (?,?,?,?,?,?,?,?)",
                (rec.frame_id, rec.ts, rec.n_detections, rec.max_conf, rec.latency_ms,
                 rec.brightness, rec.contrast, rec.edge_density),
            )
            frame_pk = cur.lastrowid
            if rec.detections:
                await self._db.executemany(
                    "INSERT INTO detections (frame_pk, ts, class_id, class_name, confidence,"
                    " x1, y1, x2, y2) VALUES (?,?,?,?,?,?,?,?,?)",
                    [(frame_pk, rec.ts, cid, cname, conf, x1, y1, x2, y2)
                     for (cid, cname, conf, x1, y1, x2, y2) in rec.detections],
                )
        await self._db.commit()

    # ----------------------------------------------------------- query API

    async def query_confidence_window(
        self, window_s: float, class_id: int | None = None
    ) -> list[float]:
        """Return detection confidences from the last ``window_s`` seconds.

        This is the feed for ConfidenceDriftDetector.check — a recent sample of
        the live confidence distribution to compare against the reference. The
        (class_id, ts) index makes the optional per-class filter cheap.
        """
        assert self._db is not None
        since = time.time() - window_s
        if class_id is None:
            sql, args = "SELECT confidence FROM detections WHERE ts >= ?", (since,)
        else:
            sql = "SELECT confidence FROM detections WHERE ts >= ? AND class_id = ?"
            args = (since, class_id)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def query_low_confidence_frames(
        self, threshold: float, window_s: float | None = None, limit: int = 100
    ) -> list[dict]:
        """Frames whose max confidence is below ``threshold`` — the AL candidates.

        These are the frames the model was least sure about; the active-learning
        queue pulls from here. Optionally restrict to a recent window and cap the
        row count so a backlog never returns an unbounded result.
        """
        assert self._db is not None
        clauses = ["max_conf < ?"]
        args: list = [threshold]
        if window_s is not None:
            clauses.append("ts >= ?")
            args.append(time.time() - window_s)
        where = " AND ".join(clauses)
        sql = (
            f"SELECT frame_id, ts, n_detections, max_conf, latency_ms "
            f"FROM frames WHERE {where} ORDER BY max_conf ASC LIMIT ?"
        )
        args.append(limit)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [
            {"frame_id": r[0], "ts": r[1], "n_detections": r[2],
             "max_conf": r[3], "latency_ms": r[4]}
            for r in rows
        ]
