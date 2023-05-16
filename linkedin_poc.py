import numpy as np
import pandas as pd


def detconstsort(items: pd.DataFrame | dict, sens_attr: str, 
                 probs: dict, kmax: int=10,):
    rankedAttList = []
    rankedScoreList = []
    rankedItems = []
    maxIndices = []
    sens_vals = list(probs.keys())
    p = list(probs.values())
    counts = np.zeros(shape=len(sens_vals), dtype=int)
    min_counts = np.zeros(shape=len(sens_vals), dtype=int).tolist()
    lastEmpty = 0
    k=0
    # check if there's enough items of each class to satisfy constraint
    for sv in probs.keys():
        target_count = kmax*probs[sv]
        if target_count > items[items[sens_attr] == sv].shape[0]:
            raise ValueError(
                f'Not enough items of group -- {sv} -- to satisfy target constraint!'
            )
    # group the items by their sensitive attribute values
    if not isinstance(items, dict):
        items = items.sort_values(axis=0, by='score',ascending=False)
        item_groups = [it for ai, it in zip(sens_vals, [items[items[sens_attr] == ai] for ai in sens_vals])]
    else:
        # check if dict is constructed correctly maybe?
        raise(NotImplementedError)
    while lastEmpty < kmax:
        k+=1
        # determine the minimum feasible counts of each group at current rec. list size
        min_counts_at_k = [int(pai*k) for pai in p]
        # get sensitive attr. values for which the current minimum count has increased
        # since last one
        changed_mins = []
        for i, ac in enumerate(zip(min_counts_at_k, min_counts)):
            min_atk = ac[0]
            min_ai = ac[1]
            if min_atk > min_ai:
                changed_mins.append(i)
            
        if len(changed_mins) > 0:
            # get the list of candidates to insert and sort them by their score
            changed_items = []
            for i in changed_mins:
                changed_items.append(item_groups[i].iloc[counts[i]])
            changed_items.sort(key=lambda x: x['score'])

            # add the items, starting with the best score
            for newitem in changed_items:
                rankedAttList.append(newitem[sens_attr])
                rankedScoreList.append(newitem['score'])
                maxIndices.append(k)
                rankedItems.append(newitem)
                start = lastEmpty
                while start > 0 and maxIndices[start-1] >= start and rankedScoreList[start-1] < rankedScoreList[start]:
                    maxIndices[start-1], maxIndices[start] = maxIndices[start], maxIndices[start-1]
                    rankedScoreList[start-1], rankedScoreList[start] = rankedScoreList[start], rankedScoreList[start-1]
                    rankedAttList[start-1], rankedAttList[start] = rankedAttList[start], rankedAttList[start-1]
                    rankedItems[start-1], rankedItems[start] = rankedItems[start], rankedItems[start-1]
                    start -= 1
                lastEmpty+=1
                counts[sens_vals.index(newitem[sens_attr])] += 1
            min_counts = min_counts_at_k
    return pd.DataFrame(rankedItems)

def gen_fake_data(sens_attr: str, sens_prop: dict, score_means: dict, k: int=100, seed: float=None):
    groups = []
    for ai in sens_prop.keys():
        df_i = pd.DataFrame(columns=['score', sens_attr])
        pi = sens_prop[ai]
        mi = score_means[ai]
        ki = int(k*pi)
        scores_i = np.random.default_rng(seed=seed).poisson(lam=mi, size=ki)
        df_i['score'] = scores_i
        df_i[sens_attr] = [ai]*ki
        groups.append(df_i)
    return pd.concat(groups)

def infeasible_index(ranking: pd.DataFrame, sens_attr: str, probs: dict, kmax: int):
    ii = 0
    for k in range(1, kmax):
        r = ranking[:k]
        for ai in probs.keys():
            count_ai = r[r[sens_attr] == ai].shape[0]
            if count_ai < int(probs[ai]*k):
                ii+=1
    return ii            


def main():
    inp = pd.DataFrame(columns = ['a', 'score'])
    df = gen_fake_data('a', {'blue': 0.7, 'red':0.3}, {'blue': 70, 'red': 40}, k=100, seed=42)
    df.sort_values(by='score', inplace=True, ascending=False)

    probs = {'red': 0.5, 'blue': 0.5}
    print(infeasible_index(df, 'a', probs, 10))
    res = detconstsort(items = df, probs = probs, kmax = 10, sens_attr = 'a')
    print(infeasible_index(res, 'a', probs, 10))
    print(pd.DataFrame(res[0]))

if __name__ == "__main__":
    main()