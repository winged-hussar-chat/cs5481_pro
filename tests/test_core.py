import numpy as np
import pandas as pd
import pytest
from src.metrics import topk,ranking_metrics
from src.data import temporal_split,targets


def test_exclusion_and_ties():
    assert topk([.2,.9,.9,.1],{1},3).tolist()==[2,0,3]
    assert topk([1,1,1],set(),2).tolist()==[0,1]
    assert len(topk([1,2],{0,1}))==0


def test_known_ranking_values():
    score=ranking_metrics([9,2,4],{2,4},10)
    assert score['Recall@10']==1
    assert score['NDCG@10']==pytest.approx((1/np.log2(3)+.5)/(1+1/np.log2(3)))
    assert ranking_metrics([9,8],{2},10)['NDCG@10']==0


def test_time_boundary_keeps_ties_together():
    frame=pd.DataFrame({'timestamp':np.repeat(np.arange(20),3)})
    train,val,test,_=temporal_split(frame)
    assert train.timestamp.max()<val.timestamp.min()
    assert val.timestamp.max()<test.timestamp.min()
    assert len(train)+len(val)+len(test)==len(frame)


def test_targets_exclude_cold_and_seen():
    b={'test':pd.DataFrame({'user_id':[1,1,1,2], 'rating':[5,4,2,5], 'item':[1,2,3,2]}),
       'histories':{1:np.array([1])},'seen':{1:{1}}}
    target,coverage=targets(b,'test')
    assert target=={1:{2}}
    assert coverage['evaluated_users']==1


def test_sequence_windows_do_not_cross_users():
    from src.train import sequence_windows
    x,y=sequence_windows({1:np.array([0,1,2]),2:np.array([5,6])},4).tensors
    assert x.tolist()==[[0,0,1,2],[0,0,0,6]]
    assert y.tolist()==[[0,0,2,3],[0,0,0,7]]


def test_lru_gradient_and_output_shape():
    import torch
    from src.models import create_lru
    model=create_lru(12,hidden=8)
    score,_=model(torch.tensor([[0,1,2,3],[0,0,4,5]]))
    assert score.shape==(2,4,13)
    assert torch.isfinite(score).all()
    score.sum().backward()
    assert model.embedding.token.weight.grad is not None
