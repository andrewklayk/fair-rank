import numpy as np
import pandas as pd


def detconstsort(items: pd.DataFrame | dict, sens_attr: str, sens_vals: list,  p: list, kmax: int):
    rankedAttList = []
    rankedScoreList = []
    rankedItems = []
    maxIndices = []
    counts = np.zeros(shape=len(sens_vals), dtype=int)
    min_counts = np.zeros(shape=len(sens_vals), dtype=int).tolist()
    lastEmpty = 0
    k=0
    # group the items by their sensitive attribute values
    if not isinstance(sens_vals, dict):
        items = items.sort_values(axis=0, by=sens_attr)
        items_by_a = [it for ai, it in zip(sens_vals, [items[items[sens_attr] == ai] for ai in sens_vals])]
    else:
        # check if dict is constructed correctly
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
                changed_items.append(items_by_a[i].iloc[counts[i]])
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
    return rankedItems, rankedScoreList, rankedAttList


def main():
    inp = pd.DataFrame(columns = ['a', 'score'])
    inp.loc[0] = ['blue', 56]
    inp.loc[1] = ['blue', 55]
    inp.loc[2] = ['blue', 45]
    inp.loc[3] = ['red', 44]
    inp.loc[4] = ['red', 44]
    inp.loc[5] = ['blue', 25]
    inp.loc[6] = ['blue', 23]
    inp.loc[7] = ['blue', 18]
    inp.loc[8] = ['red', 16]
    inp.loc[9] = ['red', 15]
    inp.loc[10] = ['red', 15]
    res = detconstsort(sens_vals = ['red', 'blue'], items = inp, p = [0.7, 0.3], kmax = 6, sens_attr = 'a')
    print(pd.DataFrame(res[0]))

if __name__ == "__main__":
    main()