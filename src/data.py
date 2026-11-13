from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
ARTIFACTS = ROOT / 'artifacts'
SEED = 42


def download():
    DATA.mkdir(exist_ok=True)
    archive = DATA / 'ml-1m.zip'
    if not (DATA / 'ml-1m/ratings.dat').exists():
        if not archive.exists():
            urllib.request.urlretrieve('https://files.grouplens.org/datasets/movielens/ml-1m.zip', archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (DATA / member.filename).resolve().is_relative_to(DATA.resolve()):
                    raise ValueError('Unsafe archive entry')
            z.extractall(DATA)
    return hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None


def read_raw():
    base = DATA / 'ml-1m'
    def read(name, cols):
        return pd.read_csv(base / name, sep='::', engine='python', names=cols, encoding='latin-1')
    ratings = read('ratings.dat', ['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = read('movies.dat', ['movie_id', 'title', 'genres'])
    users = read('users.dat', ['user_id', 'gender', 'age', 'occupation', 'zip_code'])
    return ratings, movies, users


def temporal_split(ratings):
    # All rows sharing a timestamp stay together. Globally chronological, not random.
    t1, t2 = np.quantile(ratings.timestamp, [0.8, 0.9]).astype('int64')
    return (ratings[ratings.timestamp <= t1].copy(),
            ratings[(ratings.timestamp > t1) & (ratings.timestamp <= t2)].copy(),
            ratings[ratings.timestamp > t2].copy(), (int(t1), int(t2)))


def prepare():
    checksum = download()
    ratings, movies, users = read_raw()
    train, val, test, cutoffs = temporal_split(ratings)
    catalog = np.sort(train.movie_id.unique())
    movie_map = {int(m): i for i, m in enumerate(catalog)}
    for frame in (ratings, train, val, test):
        frame['item'] = frame.movie_id.map(movie_map).fillna(-1).astype(int)
    positives = train[train.rating >= 4].sort_values(['user_id', 'timestamp', 'movie_id'])
    histories = {int(u): g.item.to_numpy() for u, g in positives.groupby('user_id')}
    seen = {int(u): set(g.item) for u, g in train.groupby('user_id')}
    bundle = dict(ratings=ratings, movies=movies, users=users, train=train, val=val, test=test,
                  catalog=catalog, histories=histories, seen=seen, cutoffs=cutoffs)
    RESULTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    stats = {'sha256': checksum, 'ratings': len(ratings), 'users': ratings.user_id.nunique(),
             'movies_in_metadata': len(movies), 'rated_movies': ratings.movie_id.nunique(),
             'train_catalog_size': len(catalog), 'splits': {k: len(bundle[k]) for k in ['train','val','test']},
             'cutoffs_utc': [str(pd.to_datetime(t, unit='s', utc=True)) for t in cutoffs],
             'relevance': 'rating >= 4', 'seed': SEED, 'duplicates': int(ratings.duplicated(['user_id','movie_id']).sum()),
             'density': len(ratings)/(ratings.user_id.nunique()*ratings.movie_id.nunique())}
    (RESULTS / 'dataset.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    summaries = []
    for name, df in [('ratings', ratings.drop(columns='item')), ('movies', movies), ('users', users)]:
        for col in df:
            summaries.append(dict(table=name, column=col, rows=len(df), dtype=str(df[col].dtype),
                                  nulls=int(df[col].isna().sum()), unique=int(df[col].nunique())))
    pd.DataFrame(summaries).to_csv(RESULTS/'data_dictionary.csv', index=False)
    return bundle


def targets(bundle, split):
    frame = bundle[split]
    relevant = frame[frame.rating >= 4]
    target = {}
    for u, rows in relevant.groupby('user_id'):
        u = int(u)
        if u not in bundle['histories']:
            continue
        items = set(rows.item) - {-1} - bundle['seen'].get(u, set())
        if items:
            target[u] = items
    coverage = dict(total_relevant_users=int(relevant.user_id.nunique()), evaluated_users=len(target),
                    unknown_item_positive_rows=int((relevant.item < 0).sum()),
                    protocol='Warm-start users and train-catalog items; all candidates, train-rated items excluded')
    return target, coverage
