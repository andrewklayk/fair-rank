import numpy as np
import pandas as pd

def dcg(scores):
    logs = np.log2(np.arange(2, len(scores)+2))
    z = np.sum(scores/logs)
    return z

def ndcg(scores):
    best = np.sort(scores)[::-1]
    return dcg(scores)/dcg(best)

def infeasible_index(ranking: pd.DataFrame, sens_attr: str, probs: dict, kmax: int):
    ii = 0
    ks = set()
    for k in range(1, kmax+1):
        r = ranking[:k]
        for ai in probs.keys():
            count_ai = r[r[sens_attr] == ai].shape[0]
            if count_ai < int(probs[ai]*k):
                ii+=1
                ks.add(k)
    return ii, ks