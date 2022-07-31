from random import choices
import numpy as np

def cleanimagename(row):
    if 'nh' in row['image']:
        return row['image'].replace('_nh','')
    elif 'nb' in row['image']:
        return row['image'].replace('_nb','')
    else:
        return row['image']
def bootstrap(dataframe1, dataframe2, variable, boot_num, stats, within=None):
    vals1 = []
    vals2 = []
    data1 = dataframe1[variable]

    if type(dataframe2 )==int:  # one frame with one value
        data2 = dataframe2
        vals2 = data2
        for i in range(boot_num):
            try:
                temp1 = choices(data1, k=len(data1))
            except:
                temp1 = choices(data1.tolist(), k=len(data1))

            if stats == "mean":
                vals1.append(np.array(temp1).mean())
            elif stats == "var":
                vals1.append(np.array(temp1).var())
        ci2 = None

    else: # two dataframe
        data2 = dataframe2[variable]
        for i in range(boot_num):
            if i % 1000 == 0:
                print(i)
            if within!=None:  # randomly sample subjects to compare within-subject difference
                dataframe1 = dataframe1[[variable, within]]
                dataframe2 = dataframe2[[variable, within]]
                dataframe1['cond'] = 'df1'
                dataframe2['cond'] = 'df2'
                subj_diff = pd.concat([dataframe1, dataframe2])
                subj_dff = subj_diff.pivot(index=within, columns='cond', values=variable).reset_index()
                subj_dff['diff'] = subj_dff['df1' ] -subj_dff['df2']
                boot_subjs = np.random.choice(np.unique(dataframe1[within]).tolist(), len(np.unique(dataframe1.subj)))
                temp1 = []
                temp2 = []
                for s in boot_subjs:
                    temp1.append(subj_dff[subj_dff[within ]==s]['diff'].item())
                    temp2.append(0)
            else:
                try:
                    temp1 = choices(data1, k=len(data1))
                except:
                    temp1 = choices(data1.tolist(), k=len(data1))
                try:
                    temp2 = choices(data2, k=len(data2))
                except:
                    temp2 = choices(data2.tolist(), k=len(data2))

            if stats == "mean":
                vals1.append(np.array(temp1).mean())
                vals2.append(np.array(temp2).mean())
            elif stats == "var":
                vals1.append(np.array(temp1).var())
                vals2.append(np.array(temp2).var())
        ci2 = np.percentile(vals2, [2.5 ,97.5])  # 95% ci

    ci1 = np.percentile(vals1, [2.5 ,97.5])  # 95% ci

    try:
        p = 1 - len([1 for (i ,j) in zip(vals1, vals2) if i< j]) / boot_num
    except:
        p = 1 - len([1 for i in vals1 if i < data2]) / boot_num
    p = np.min([p, 1 - p])
    return ci1, ci2, p

