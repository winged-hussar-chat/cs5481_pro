import numpy as np
import pytest
from src.data import ARTIFACTS


@pytest.mark.skipif(not (ARTIFACTS/'bundle.joblib').exists(),reason='Train models before integration tests')
def test_real_recommendations_exclude_every_rated_movie():
    from src.service import RecommendationService
    service=RecommendationService()
    users=sorted(service.bundle['histories'])
    for user in [users[0],users[len(users)//2],users[-1]]:
        seen=set(service.bundle['ratings'].query('user_id == @user').movie_id)
        for model in ['ItemCF','LightGBM','LRURec']:
            result=service.recommend(user,model,10)
            assert len(result)==10
            assert result.movie_id.nunique()==10
            assert not (set(result.movie_id)&seen)
            assert np.isfinite(result.score).all()
            filtered=service.recommend(user,model,5,'Comedy')
            assert filtered.genres.str.contains('Comedy').all()


def test_feature_inputs_remain_training_only():
    import pandas as pd
    from src.models import RetailFeatures
    context=pd.DataFrame({'user_id':[1,1],'item':[0,1],'rating':[4,5]})
    bundle={'catalog':np.array([10,20,30]),
            'movies':pd.DataFrame({'movie_id':[10,20,30],'title':['A (1990)','B (1991)','C'], 'genres':['Comedy','Drama','Comedy']}),
            'users':pd.DataFrame({'user_id':[1],'age':[25],'gender':['M'],'occupation':[1]})}
    features=RetailFeatures().fit(bundle,context)
    x=features.transform([1,1,1],[0,1,2])
    assert np.isfinite(x).all()
    assert np.all((x[:,1]>=0)&(x[:,1]<=1))
    assert features.item_count[2]==0
