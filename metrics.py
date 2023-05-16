import numpy as np

def dcg(scores):
    logs = np.log2(np.arange(2, len(scores)+2))
    z = np.sum(scores/logs)
    return z

def ndcg(scores):
    best = np.sort(scores)[::-1]
    return dcg(scores)/dcg(best)