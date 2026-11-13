import numpy as np


def topk(scores, seen=(), k=10):
    scores = np.asarray(scores, dtype=float).copy()
    if len(seen):
        scores[np.asarray(list(seen), dtype=int)] = -np.inf
    eligible = np.flatnonzero(np.isfinite(scores))
    # Stable item-id tie breaking makes repeated runs reproducible.
    return eligible[np.lexsort((eligible, -scores[eligible]))[:k]]


def ranking_metrics(ranked, relevant, k=10):
    if not relevant:
        raise ValueError('Relevant set must not be empty')
    hits = np.array([i in relevant for i in ranked[:k]], dtype=float)
    dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
    ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return {'Recall@10': float(hits.sum()/len(relevant)), 'NDCG@10': float(dcg/ideal),
            'HitRate@10': float(hits.any())}
