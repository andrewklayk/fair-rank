import numpy as np
import pandas as pd

def detgreedy(items: pd.DataFrame, s: str, 
                 props: dict, kmax: int=10):
    rankedItems = []
    counts_a = {a: 0 for a in props.keys()}
    #counts_a = np.zeros(shape=len(probs.keys()))
    
    items = items.sort_values(axis=0, by='score',ascending=False)
    item_groups = {ai: it for ai, it in zip(props.keys(), [items[items[s] == ai] for ai in props.keys()])}

    for k in range(1,kmax):
        below_min = []
        below_max = []
        candidates = [candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in item_groups.items()]
        for ai in props.keys():
            # best unranked items for each sensitive attribute
            if counts_a[ai] < np.floor(k*props[ai]):
                below_min.append((ai))
            elif counts_a[ai] < np.ceil(k*props[ai]):
                below_max.append((ai))
        if len(below_min) != 0:
            candidates_bmin = [c for c in candidates if c[s] in below_min]
            next_item = max(candidates_bmin, key = lambda x: x['score'])
        else:
            candidates_bmax = [c for c in candidates if c[s] in below_max]
            next_item = max(candidates_bmax, key = lambda x: x['score'])
        rankedItems.append(next_item)
        counts_a[next_item[s]] += 1
    return pd.DataFrame(rankedItems)

def detcons(items: pd.DataFrame, s: str, 
                 props: dict, kmax: int=10, relaxed=False):
    rankedItems = []
    counts_a = {a: 0 for a in props.keys()}
    #counts_a = np.zeros(shape=len(probs.keys()))
    
    items = items.sort_values(axis=0, by='score',ascending=False)
    item_groups = {ai: it for ai, it in zip(props.keys(), [items[items[s] == ai] for ai in props.keys()])}

    for k in range(1,kmax):
        below_min = []
        below_max = []
        candidates = [candidates_ai.iloc[counts_a[ai]] for ai, candidates_ai in item_groups.items()]
        for ai in props.keys():
            # best unranked items for each sensitive attribute
            if counts_a[ai] < np.floor(k*props[ai]):
                below_min.append((ai))
            elif counts_a[ai] < np.ceil(k*props[ai]):
                below_max.append((ai))
        if len(below_min) != 0:
            candidates_bmin = [c for c in candidates if c[s] in below_min]
            next_item = max(candidates_bmin, key = lambda x: x['score'])
        else:
            # sort by scores if tie in lambda?
            if relaxed:
                next_attr_set = min(below_max, key=lambda ai: np.ceil(np.ceil(k*props[ai])/props[ai]))
                if not isinstance(next_attr_set, list):
                    next_attr_set = [next_attr_set]
                candidates_rel = [c for c in candidates if c[s] in next_attr_set]
                # best item among best items for each attribute in next_attr_set
                next_item = max(candidates_rel, key=lambda x: x['score'])
            else:
                next_attr = min(below_max, key=lambda ai: np.ceil(k*props[ai])/props[ai])
                next_item = item_groups[next_attr].iloc[counts_a[next_attr]]
        rankedItems.append(next_item)
        counts_a[next_item[s]] += 1
    return pd.DataFrame(rankedItems)

def detconstsort(items: pd.DataFrame | dict, s: str, 
                 props: dict, kmax: int=10,):
    rankedAttList = []
    rankedScoreList = []
    rankedItems = []
    maxIndices = []
    sens_vals = list(props.keys())
    p = list(props.values())
    counts = np.zeros(shape=len(sens_vals), dtype=int)
    min_counts = np.zeros(shape=len(sens_vals), dtype=int).tolist()
    lastEmpty = 0
    k=0
    # check if there's enough items of each class to satisfy constraint
    for sv in props.keys():
        target_count = kmax*props[sv]
        if target_count > items[items[s] == sv].shape[0]:
            raise ValueError(
                f'Not enough items of group -- {sv} -- to satisfy target constraint!'
            )
    # group the items by their sensitive attribute values
    if not isinstance(items, dict):
        items = items.sort_values(axis=0, by='score',ascending=False)
        item_groups = [it for ai, it in zip(sens_vals, [items[items[s] == ai] for ai in sens_vals])]
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
                rankedAttList.append(newitem[s])
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
                counts[sens_vals.index(newitem[s])] += 1
            min_counts = min_counts_at_k
    return pd.DataFrame(rankedItems)