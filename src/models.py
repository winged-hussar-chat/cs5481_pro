import importlib.util
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from lightgbm import LGBMClassifier
from .data import ROOT, SEED


class ItemCF:
    def __init__(self, k=80):
        self.k = k

    def fit(self, train, n_items):
        positives = train[train.rating >= 4]
        matrix = csr_matrix((np.ones(len(positives), dtype=np.float32),
                            (positives.user_id.to_numpy(), positives.item.to_numpy())),
                            shape=(6041, n_items))
        norm = normalize(matrix.T, axis=1)
        sim = (norm @ norm.T).toarray()
        np.fill_diagonal(sim, 0)
        k = min(self.k, n_items)
        keep = np.argpartition(sim, -k, axis=1)[:, -k:]
        rows = np.repeat(np.arange(n_items), k)
        self.similarity = csr_matrix((sim[rows, keep.ravel()], (rows, keep.ravel())), shape=sim.shape)
        return self

    def score(self, history):
        if not len(history):
            return np.zeros(self.similarity.shape[0], dtype=np.float32)
        return np.asarray(self.similarity[np.unique(history)].sum(axis=0)).ravel()


class RetailFeatures:
    """Fixed inner-training features: no label-row or held-out statistics."""
    def fit(self, bundle, context):
        n = len(bundle['catalog'])
        movies = bundle['movies'].set_index('movie_id').loc[bundle['catalog']]
        self.genre_names = sorted(set('|'.join(movies.genres).split('|')))
        self.genres = np.array([[g in s.split('|') for g in self.genre_names] for s in movies.genres], dtype=np.float32)
        self.year = movies.title.str.extract(r'((d{4}))$')[0].fillna('1995').astype(float).to_numpy()/2000
        self.item_count = np.zeros(n, dtype=np.float32)
        self.item_mean = np.full(n, context.rating.mean()/5, dtype=np.float32)
        agg = context.groupby('item').rating.agg(['count', 'mean'])
        self.item_count[agg.index] = np.log1p(agg['count'])
        self.item_mean[agg.index] = agg['mean']/5
        self.profiles = np.zeros((6041, len(self.genre_names)), dtype=np.float32)
        self.user_count = np.zeros(6041, dtype=np.float32)
        self.user_mean = np.zeros(6041, dtype=np.float32)
        for u, group in context.groupby('user_id'):
            self.user_count[u] = np.log1p(len(group))
            self.user_mean[u] = group.rating.mean()/5
            liked = group[group.rating >= 4].item.to_numpy()
            if len(liked): self.profiles[u] = self.genres[liked].mean(axis=0)
        self.demographics = np.zeros((6041, 3), dtype=np.float32)
        users = bundle['users']
        self.demographics[users.user_id] = np.c_[users.age/56, (users.gender=='M').astype(float), users.occupation/20]
        return self

    def transform(self, users, items):
        users, items = np.asarray(users), np.asarray(items)
        affinity = self.profiles[users] * self.genres[items]
        return np.c_[self.item_count[items], self.item_mean[items], self.year[items],
                     self.user_count[users], self.user_mean[users], self.demographics[users],
                     self.genres[items], affinity, affinity.sum(axis=1)].astype(np.float32)


class LightGBMRec:
    def fit(self, bundle, leaves=31):
        train = bundle['train']
        cutoff = int(train.timestamp.quantile(.8))
        context, labels = train[train.timestamp <= cutoff], train[train.timestamp > cutoff]
        self.features = RetailFeatures().fit(bundle, context)
        positives = labels[labels.rating >= 4]
        self.n_items = len(bundle['catalog'])
        rng = np.random.default_rng(SEED)
        uu, ii, yy = [], [], []
        train_seen = {int(u):set(g.item) for u,g in train.groupby('user_id')}
        for u, group in positives.groupby('user_id'):
            p = group.item.to_numpy()
            available = np.setdiff1d(np.arange(self.n_items), list(train_seen[u]))
            negative = rng.choice(available, min(len(available), len(p)*4), replace=False)
            items = np.r_[p, negative]
            uu.extend([int(u)]*len(items)); ii.extend(items); yy.extend([1]*len(p)+[0]*len(negative))
        self.model = LGBMClassifier(n_estimators=220, num_leaves=leaves, learning_rate=.05,
                                   min_child_samples=40, reg_lambda=1, random_state=SEED,
                                   n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True)
        self.model.fit(self.features.transform(uu, ii), yy)
        self.training_info = dict(rows=len(yy), positives=int(sum(yy)), inner_cutoff=cutoff,
                                  negatives_per_positive=4, feature_count=self.model.n_features_in_)
        return self

    def score(self, user):
        items = np.arange(self.n_items)
        return self.model.predict_proba(self.features.transform(np.full(self.n_items,user), items))[:,1]


def create_lru(n_items, hidden=32, blocks=2, dropout=.2):
    path = ROOT/'vendor/LRURec/model/lru.py'
    if not path.exists():
        raise RuntimeError('Run git submodule update --init --recursive first')
    spec = importlib.util.spec_from_file_location('upstream_lru', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = SimpleNamespace(num_items=n_items, bert_hidden_units=hidden, bert_num_blocks=blocks,
                           bert_dropout=dropout, bert_attn_dropout=dropout, dataset_code='ml-1m')
    return mod.LRU(args)
