import pandas as pd
import glob, os
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multitest import multipletests as mt
from itertools import combinations
from scipy import stats
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
from evaluate_models.utils import *
setpallet = sns.color_palette("Set2")
custom_colors = sns.color_palette("Set1", 10)


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
summaries = glob.glob('data/GroundTruth_humanest/*summary*')
results = pd.DataFrame()
for f in summaries:
    data = pd.read_excel(f)
    results = results.append(data)
results['test_cond'] = results['test_cond'].astype('category')
results['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
results['model'] = results['model'].astype('category')
results['model'].cat.reorder_categories(['Humans', 'CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer'], inplace=True)





''' PART I Human, CNN, 3 Transformer performance (wrt human estimates), which model is more consistent with humans '''
test_cond = 'headless bodies'
plot_data = results[results['test_cond']==test_cond]
plot_data = plot_data[[ 'Euclidean_error_meanest', 'Angular_error_meanest',
       'Euclidean_error_lou', 'Angular_error_lou','model']]

error = 'Euclidean_error_meanest'
aov_data = plot_data[[error, 'model']].melt(id_vars='model')
aov = pg.anova(dv='value', between=['model'], data=aov_data,
             detailed=True)
print(aov)
postdoc =aov_data.pairwise_ttests(dv='value',
                                   between=['model'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = postdoc[postdoc['p-corr']<0.05]
box_pairs = []
ps = []
for _, row in sig_results.iterrows():
    box_pairs.append((row['A'],row['B']))
    ps.append(max(0.001, row['p-corr']))


error = 'Euclidean_error_meanest'
sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y =  error,color=setpallet[0])
ax.set(xlabel='', ylabel='Euclidean Error')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
add_stat_annotation(ax, data=plot_data, x = 'model', y =  error,
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
plt.xticks(rotation=90, fontsize=20)
ax.figure.savefig("figures/{}_gt_humanest_{}.png".format(test_cond, error), dpi=300, bbox_inches='tight')
plt.close()



# error = 'Euclidean_error_lou'
# sns_setup_small(sns, (8,6))
# ax = sns.barplot(data = plot_data, x = 'model', y = error, color=setpallet[0])
# ax.set(xlabel='', ylabel='Euclidean Error')
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# add_stat_annotation(ax, data=plot_data, x = 'model', y = error,
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside', verbose=2)
# plt.xticks(rotation=90, fontsize=20)
# ax.figure.savefig("figures/{}_gt_humanest_{}.png".format(test_cond,error), dpi=300, bbox_inches='tight')
# plt.close()



error = 'Angular_error_meanest'
sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y = error,color=setpallet[1])
ax.set(xlabel='', ylabel='Angular Error (˚)')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
add_stat_annotation(ax, data=plot_data, x = 'model', y = error,
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
plt.xticks(rotation=90, fontsize=20)
ax.figure.savefig("figures/{}_gt_humanest_{}.png".format(test_cond, error), dpi=300, bbox_inches='tight')
plt.close()



error = 'Angular_error_lou'
sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y =  error,color=setpallet[1])
ax.set(xlabel='', ylabel='Angular Error (˚)')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
add_stat_annotation(ax, data=plot_data, x = 'model', y = error,
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
plt.xticks(rotation=90, fontsize=20)
ax.figure.savefig("figures/{}_gt_humanest_{}.png".format(test_cond,error), dpi=300, bbox_inches='tight')
plt.close()




''' PART II Human-Human, Human-CNN, Human-Transformer Correlation on Euclidean and Angular Error '''
# calculate human-human error wrt mean gaze estimation or leave-one-out estimation

humans = pd.read_excel('data/GroundTruth_humanest/Humans_summary.xlsx')
humans = humans[humans['test_cond']=='intact']
subjects = list(np.unique(humans['subj']))
# subj1, subj2, euc_meanest_error, ang_meanest_error, euc_lou_error, ang_lou_error = [], [], [], [], [], []
# for s1 in subjects:
#     print(s1)
#     rest_subjects = subjects.copy()
#     rest_subjects.remove(s1)
#     for s2 in rest_subjects:
#         # subjs = random.sample(list(subjects), 2)
#         tempdata = humans[(humans['subj']==s1) | (humans['subj']==s2)]
#         tempdata = tempdata[['image','subj','Euclidean_error_meanest','Angular_error_meanest',
#                              'Euclidean_error_lou','Angular_error_lou']]
#         tempdata= tempdata.pivot(index=["image"], columns=["subj"]).dropna().reset_index()
#         tempdata.columns = tempdata.columns.droplevel(1)
#         tempdata.columns = ['image', 'Euclidean_meanest_subj1', 'Euclidean_meanest_subj2',
#                                     'Angular_meanest_subj1', 'Angular_meanest_subj2',
#                                     'Euclidean_lou_subj1', 'Euclidean_lou_subj2',
#                                     'Angular_lou_subj1', 'Angular_lou_subj2'
#                             ]
#         subj1.append(s1)
#         subj2.append(s2)
#         r, p = stats.pearsonr(tempdata["Euclidean_meanest_subj1"], tempdata["Euclidean_meanest_subj2"])
#         euc_meanest_error.append(r)
#         r, p = stats.pearsonr(tempdata["Angular_meanest_subj1"], tempdata["Angular_meanest_subj2"])
#         ang_meanest_error.append(r)
#         r, p = stats.pearsonr(tempdata["Euclidean_lou_subj1"], tempdata["Euclidean_lou_subj2"])
#         euc_lou_error.append(r)
#         r, p = stats.pearsonr(tempdata["Angular_lou_subj1"], tempdata["Angular_lou_subj2"])
#         ang_lou_error.append(r)
#
# humans_humans = pd.DataFrame({'subj1':subj1, 'subj2':subj2,
#                            'Euclidean Error Mean':euc_meanest_error, 'Angular Error Mean':ang_meanest_error,
#                            'Euclidean Error LOU':euc_lou_error, 'Angular Error LOU':ang_lou_error})
# humans_humans.to_excel('data/GroundTruth_humanest/Human_intact_error_corr.xlsx',index=None)

# 1. human-human correlation
humans_humans = pd.read_excel('data/GroundTruth_humanest/Human_intact_error_corr.xlsx')
humans_humans['corr_rel'] = 'Humans-Humans'


# 2. human-model correlation
intact = results[(results['test_cond']=='intact') & (results['model']!='Humans')]
models = ['CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']
humans_models = pd.DataFrame()
for model in models:
    model_data = intact[(intact['model']==model)][['image','Euclidean_error_meanest', 'Angular_error_meanest',
       'Euclidean_error_lou', 'Angular_error_lou', 'model']]
    subj1, euc_error_meanest, ang_error_meanest, euc_error_lou, ang_error_lou = [], [], [], [], []
    for s in subjects: #(individual subject & CNN)
        s_data = humans[humans['subj']==s][['image','Euclidean_error_meanest', 'Angular_error_meanest',
       'Euclidean_error_lou', 'Angular_error_lou', 'model']]
        subj1.append(s)
        humans_model = pd.concat([s_data, model_data])
        try:
            plot_data_piv = humans_model.pivot(index=["image"], columns=["model"], ).dropna().reset_index()
            plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
            plot_data_piv.columns = ['image','Euclidean_meanest_model', 'Euclidean_meanest_Humans',
                                     'Angular_meanest_model', 'Angular_meanest_Humans',
                                     'Euclidean_lou_model', 'Euclidean_lou_Humans',
                                     'Angular_lou_model', 'Angular_lou_Humans',
                                     ]
        except:
            plot_data_piv = humans_model.pivot_table(index='image', columns='model', aggfunc='mean').dropna().reset_index()
            plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
            plot_data_piv.columns = ['image','Angular_lou_model', 'Angular_lou_Humans',
                                     'Angular_meanest_model', 'Angular_meanest_Humans',
                                     'Euclidean_lou_model', 'Euclidean_lou_Humans',
                                     'Euclidean_meanest_model', 'Euclidean_meanest_Humans']
        eu_r, eu_p = stats.pearsonr(plot_data_piv["Euclidean_meanest_model"], plot_data_piv["Euclidean_meanest_Humans"])
        euc_error_meanest.append(eu_r)
        ang_r, ang_p = stats.pearsonr(plot_data_piv["Angular_meanest_model"], plot_data_piv["Angular_meanest_Humans"])
        ang_error_meanest.append(ang_r)

        eu_r, eu_p = stats.pearsonr(plot_data_piv["Euclidean_lou_model"], plot_data_piv["Euclidean_lou_Humans"])
        euc_error_lou.append(eu_r)
        ang_r, ang_p = stats.pearsonr(plot_data_piv["Angular_lou_model"], plot_data_piv["Angular_lou_Humans"])
        ang_error_lou.append(ang_r)

    humans_model = pd.DataFrame({'subj1':subj1, 'subj2':[model]*len(subj1),
                                 'Euclidean Error Mean':euc_error_meanest, 'Angular Error Mean':ang_error_meanest,
                                 'Euclidean Error LOU':euc_error_lou, 'Angular Error LOU':ang_error_lou})
    humans_model['corr_rel'] = 'Humans-{}'.format(model)
    humans_models = humans_models.append(humans_model, ignore_index=True)


gttype = 'LOU' #LOU/Mean
all_corr = pd.concat([humans_humans, humans_models])
plot_data = all_corr[['Euclidean Error {}'.format(gttype),'Angular Error {}'.format(gttype), 'corr_rel']]
plot_data = plot_data.melt(id_vars=['corr_rel'])
sns_setup_small(sns, (8,6))
ax = sns.barplot(data=plot_data, x='value', y='corr_rel', hue='variable')
ax.set(xlabel='Correlation',ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(frameon=False)
ax.figure.savefig("figures/intact_gt_humanest_{}_allcorr.png".format(gttype), dpi=300, bbox_inches='tight')
plt.close()




''' PART III Human-Human, Human-CNN, Human-Transformer Vector Angle Correlation '''
files = glob.glob('data/GroundTruth_humanest/*vectors.xlsx')
results = pd.DataFrame()
for f in files:
    df = pd.read_excel(f)
    df.columns = [x if 'est' not in x else '_'.join(x.split('_')[1:]) for x in df.columns ]
    results = results.append(df, ignore_index=True)
results = results[['cond','image','gazer','subj','Angle2Hori','model']]

# 1. human-human correlation
humans = results[(results['cond']=='intact') & (results['model']=='Humans')]
humans = humans.groupby(['cond','image','model','subj']).mean().reset_index().drop('gazer',axis=1)
subjects = list(np.unique(humans['subj']))
# subj1, subj2, corr = [], [], []
# for s1 in subjects:
#     print(s1)
#     rest_subjects = subjects.copy()
#     rest_subjects.remove(s1)
#     for s2 in rest_subjects:
#         tempdata = humans[(humans['subj']==s1) | (humans['subj']==s2)]
#         tempdata = tempdata[['image','subj','Angle2Hori']]
#         tempdata= tempdata.pivot(index=["image"], columns=["subj"]).dropna().reset_index()
#         tempdata.columns = tempdata.columns.droplevel(1)
#         tempdata.columns = ['image', 'Angle2Hori_subj1', 'Angle2Hori_subj2']
#         subj1.append(s1)
#         subj2.append(s2)
#         r, p = stats.pearsonr(tempdata["Angle2Hori_subj1"], tempdata["Angle2Hori_subj2"])
#         corr.append(r)
#
# human_corr = pd.DataFrame({'subj1':subj1, 'subj2':subj2, 'vec_angle_corr':corr})
# human_corr['corr_rel'] = 'Humans-Humans'
# human_corr.to_excel('data/GroundTruth_humanest/Human_intact_vec_angle_corr.xlsx',index=None)
humans_humans = pd.read_excel('data/GroundTruth_humanest/Human_intact_vec_angle_corr.xlsx')

# 2. human-model correlation
allmodels = results[(results['cond']=='intact') & (results['model']!='Humans')].drop('subj',axis=1)
allmodels = allmodels.groupby(['cond','image','model']).mean().reset_index().drop('gazer',axis=1)

models = ['CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']
humans_models = pd.DataFrame()
for model in models:
    model_data = allmodels[(allmodels['model']==model)].drop('cond',axis=1)
    subj1, corr = [], []
    for s in subjects: #(individual subject & CNN)
        s_data = humans[humans['subj']==s][['image','Angle2Hori','model']]
        subj1.append(s)
        humans_model = pd.concat([s_data, model_data])
        plot_data_piv = humans_model.pivot(index=["image"], columns=["model"]).dropna().reset_index()
        plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
        plot_data_piv.columns = ['image','Angle2Hori_model', 'Angle2Hori_Humans']
        r, p = stats.pearsonr(plot_data_piv["Angle2Hori_model"], plot_data_piv["Angle2Hori_Humans"])
        corr.append(r)

    humans_model = pd.DataFrame({'subj1':subj1, 'subj2':[model]*len(subj1),'vec_angle_corr': corr})
    humans_model['corr_rel'] = 'Humans-{}'.format(model)
    humans_models = humans_models.append(humans_model, ignore_index=True)


# plot
all_corr = pd.concat([humans_humans, humans_models])
plot_data = all_corr[['vec_angle_corr','corr_rel']]
plot_data = plot_data.melt(id_vars=['corr_rel'])


# bootstrap
boot_data = all_corr
boot_data = boot_data.groupby(['corr_rel','subj1']).mean().reset_index()
cond = list(np.unique(boot_data.corr_rel))
conditions = set(list(combinations(cond, 2)))
pvals = pd.DataFrame()
for var in ['vec_angle_corr']:
    for cond1, cond2 in conditions:
        print(cond1)
        print(cond2)
        dataframe1 = boot_data[boot_data['corr_rel']==cond1]
        dataframe2 = boot_data[boot_data['corr_rel'] == cond2]
        ci1, ci2, p = bootstrap(dataframe1, dataframe2, var, 10000, 'mean')
        pvals = pvals.append({'variable':var, 'cond1': cond1, 'cond2': cond2,
                                  'ci1l':ci1[0], 'ci1u':ci1[1], 'ci2l':ci2[0], 'ci2u':ci2[1],
                                  'p': p}, ignore_index=True)

p_adjs = mt(pvals['p'], alpha=0.05, method='fdr_bh')[1]
pvals['p_adj'] = p_adjs
# pvals.to_excel('data/boot_results.xlsx',index=None)
sig_pvals = pvals[pvals['p_adj']<0.05]
if len(sig_pvals)>0:
    ps = list(sig_pvals['p_adj'])
box_pairs = []
for _, row in sig_pvals.iterrows():
    cond1, cond2 = row['cond1'], row['cond2']
    var = row['variable']
    box_pairs.append((cond1, cond2))


sns_setup_small(sns, (8,6))
ax = sns.barplot(data=plot_data, x= 'corr_rel', y='value' ,color=setpallet[2])
ax.set(xlabel='',ylabel='Correlation') #,title='Vector Angle Correlation'
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(frameon=False)
add_stat_annotation(ax, data=plot_data, x='corr_rel', y='value',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
plt.xticks(rotation=90, fontsize=20)
ax.figure.savefig("figures/intact_gt_humanest_vec_ang_corr.png", dpi=300, bbox_inches='tight')
plt.close()
