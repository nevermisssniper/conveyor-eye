from scripts.demo import report

def test_threshold_extremes_and_reproducibility():
    a=report(0);b=report(1)
    assert a['recall']==1 and a['fn']==0
    assert b['tp']==0 and b['fp']==0
    assert report(.5)==report(.5)
    assert len(a['samples'])==120
