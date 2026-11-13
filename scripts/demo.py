"""Seeded operating-threshold demo. No detector weights or inference claims."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from conveyoreye.monitoring.drift import population_stability_index

def report(threshold=.5,seed=42):
    if not 0<=threshold<=1:raise ValueError('threshold must be in [0,1]')
    rng=np.random.default_rng(seed);truth=rng.random(120)<.3
    scores=np.where(truth,rng.beta(5,2,120),rng.beta(2,5,120));accepted=scores>=threshold
    tp=int((accepted&truth).sum());fp=int((accepted&~truth).sum());fn=int((~accepted&truth).sum())
    return {'mode':'synthetic_scores','threshold':threshold,'tp':tp,'fp':fp,'fn':fn,
      'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
      'shift_psi':float(population_stability_index(scores,np.clip(scores-.15,0,1),np.linspace(0,1,11))),
      'samples':[{'confidence':float(s),'defective':bool(t)} for s,t in zip(scores,truth)]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--threshold',type=float,default=.5);p.add_argument('--output');a=p.parse_args();text=json.dumps(report(a.threshold),indent=2)
    if a.output:Path(a.output).write_text(text+'\n')
    else:print(text)
