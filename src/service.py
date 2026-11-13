import joblib
import numpy as np
import torch
from .data import ARTIFACTS
from .metrics import topk
from .models import create_lru


class RecommendationService:
    def __init__(self):
        torch.set_num_threads(4)
        self.bundle = joblib.load(ARTIFACTS/'bundle.joblib')
        self.cf = joblib.load(ARTIFACTS/'itemcf.joblib')
        self.lgb = joblib.load(ARTIFACTS/'lightgbm.joblib')
        ckpt = torch.load(ARTIFACTS/'lrurec.pt',map_location='cpu',weights_only=True)
        self.max_len = ckpt['max_len']
        self.lru = create_lru(ckpt['n_items'],ckpt['hidden'])
        self.lru.load_state_dict(ckpt['state_dict']); self.lru.eval()
        self.movies = self.bundle['movies'].set_index('movie_id')

    def scores(self,user,model):
        history=self.bundle['histories'][user]
        if model=='ItemCF': return self.cf.score(history)
        if model=='LightGBM': return self.lgb.score(user)
        seq=np.zeros((1,self.max_len),dtype=np.int64)
        h=history[-self.max_len:]+1; seq[0,-len(h):]=h
        with torch.no_grad():
            return self.lru(torch.from_numpy(seq))[0][0,-1,1:].numpy()

    def recommend(self,user,model,k=10,genre='All genres'):
        if user not in self.bundle['histories']:
            raise ValueError('Select a user with positive training history')
        scores=self.scores(user,model).copy()
        # Demo excludes ALL recorded ratings. Evaluation uses train-only exclusions.
        rated=self.bundle['ratings'].query('user_id == @user').item
        seen=set(rated[rated>=0])
        if genre!='All genres':
            allowed=self.movies.loc[self.bundle['catalog']].genres.str.split('|').map(lambda gs:genre in gs).to_numpy()
            scores[~allowed]=-np.inf
        items=topk(scores,seen,k)
        frame=self.movies.loc[self.bundle['catalog'][items]].reset_index()
        frame.insert(0,'rank',np.arange(1,len(frame)+1))
        frame['score']=scores[items]
        return frame
