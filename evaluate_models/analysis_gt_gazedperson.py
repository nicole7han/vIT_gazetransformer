import pandas as pd
import glob, os
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from scipy import stats
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")
custom_colors = sns.color_palette("Set1", 10)


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
summaries = glob.glob('data/GroundTruth_gazedperson/*summary*')
results = pd.DataFrame()
for f in summaries:
    data = pd.read_excel(f)
    results = results.append(data)
results['test_cond'] = results['test_cond'].astype('category')
results['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
results['model'] = results['model'].astype('category')
results['model'].cat.reorder_categories(['Humans', 'CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer'], inplace=True)


''' Human, CNN, 3 Transformer performance (human estimates as groundtruth) '''
plot_data = results[results['test_cond']=='intact']
plot_data = plot_data[[ 'Euclidean_error', 'Angular_error','model']]

error = 'Euclidean_error'
# aov_data = plot_data[[error, 'model']].melt(id_vars='model')
# aov = pg.anova(dv='value', between=['model'], data=aov_data,
#              detailed=True)
# print(aov)
# postdoc =aov_data.pairwise_ttests(dv='value',
#                                    between=['model'],
#                                    padjust='fdr_bh',
#                                    parametric=True).round(3)
# sig_results = postdoc[postdoc['p-corr']<0.05]
# box_pairs = []
# ps = []
# for _, row in sig_results.iterrows():
#     box_pairs.append((row['A'],row['B']))
#     ps.append(max(0.001, row['p-corr']))

sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[0])
ax.set(xlabel='Euclidean Error', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig("figures/intact_gt_gazedperson_{}.png".format(error), dpi=300, bbox_inches='tight')
plt.close()



sns_setup_small(sns, (8,6))
error = 'Angular_error'
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[1])
ax.set(xlabel='Angular Error (˚)', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig("figures/intact_gt_gazedperson_{}.png".format(error), dpi=300, bbox_inches='tight')
plt.close()





''' Human-Human, Human-CNN, Human-Transformer Correlation on Euclidean and Angular Error '''
# calculate human-human error wrt gazed person
image_info = pd.read_excel('data/GroundTruth_gazedperson/image_info.xlsx')
human_path = '/Users/nicolehan/Documents/Research/GazeExperiment/Mechanical turk/Analysis_absent'
# results = glob.glob('{}/human*.xlsx'.format(human_path))
# humans = pd.DataFrame()
# for f in results:
#     df = pd.read_excel(f)
#     df.columns = ['human_est_x', 'human_est_y', 'subj', 'condition', 'movie', 'image']
#     if 'intact' in f: Test_cond = 'intact'
#     elif 'floating heads' in f: Test_cond = 'floating heads'
#     elif 'headless bodies' in f: Test_cond = 'headless bodies'
#     df = df.drop(['condition','movie'],axis=1)
#     df = df.merge(image_info, on=['image'])
#     df['Euclidean_error'] = np.sqrt(
#         (df['gazed_x'] - df['human_est_x']) ** 2 + (df['gazed_y'] - df['human_est_y']) ** 2)
#     df['Angular_error'] = df.apply(lambda r: compute_angle(r,'human'),axis=1)
#     df['test_cond'] = Test_cond
#     humans = pd.concat([humans,df])
# humans = humans[(humans['subj']!=99401) & (humans['subj']!=99807)]
# humans = humans.groupby(['subj','image','test_cond']).mean().reset_index() # average across gazers
# humans.to_excel('data/GroundTruth_gazedperson/Human_estimations.xlsx',index=None)

# humans = pd.read_excel('data/GroundTruth_gazedperson/Human_estimations.xlsx')
# humans = humans[humans['test_cond']=='intact']
# subjects = list(np.unique(humans['subj']))
# subj1, subj2, euc_error, ang_error = [], [], [], []
# for s1 in subjects:
#     print(s1)
#     rest_subjects = subjects.copy()
#     rest_subjects.remove(s1)
#     for s2 in rest_subjects:
#         # subjs = random.sample(list(subjects), 2)
#         tempdata = humans[(humans['subj']==s1) | (humans['subj']==s2)]
#         tempdata = tempdata[['image','subj','Euclidean_error','Angular_error']]
#         tempdata= tempdata.pivot(index=["image"], columns=["subj"]).dropna().reset_index()
#         tempdata.columns = tempdata.columns.droplevel(1)
#         tempdata.columns = ['image', 'Euclidean_subj1', 'Euclidean_subj2', 'Angular_subj1', 'Angular_subj2']
#         subj1.append(s1)
#         subj2.append(s2)
#         r, p = stats.pearsonr(tempdata["Euclidean_subj1"], tempdata["Euclidean_subj2"])
#         euc_error.append(r)
#         r, p = stats.pearsonr(tempdata["Angular_subj1"], tempdata["Angular_subj2"])
#         ang_error.append(r)
#
# human_corr = pd.DataFrame({'subj1':subj1, 'subj2':subj2, 'Euclidean Error':euc_error, 'Angular Error':ang_error})
# human_corr.to_excel('data/GroundTruth_gazedperson/Human_intact_error_corr.xlsx',index=None)

## Individual human estimation errors
humans = pd.read_excel('data/GroundTruth_gazedperson/Human_estimations.xlsx')
humans = humans[humans['test_cond']=='intact']
humans['model'] = 'Humans'
subjects = list(np.unique(humans['subj']))

# 1. human-human correlation
humans_humans = pd.read_excel('data/GroundTruth_gazedperson/Human_intact_error_corr.xlsx')
humans_humans['corr_rel'] = 'Human-Human'


# 2. human-model correlation
intact = results[(results['test_cond']=='intact') & (results['model']!='Humans')]
models = ['CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']
humans_models = pd.DataFrame()
for model in models:
    model_data = intact[(intact['model']==model)].drop('test_cond',axis=1)
    subj1, euc_error, ang_error = [], [], []
    for s in subjects: #(individual subject & CNN)
        s_data = humans[humans['subj']==s][['image','Euclidean_error','Angular_error','model']]
        subj1.append(s)
        humans_model = pd.concat([s_data, model_data])
        plot_data_piv = humans_model.pivot(index=["image"], columns=["model"]).dropna().reset_index()
        plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
        plot_data_piv.columns = ['image','Euclidean_model', 'Euclidean_Humans', 'Angular_model', 'Angular_Humans']
        eu_r, eu_p = stats.pearsonr(plot_data_piv["Euclidean_model"], plot_data_piv["Euclidean_Humans"])
        euc_error.append(eu_r)
        ang_r, ang_p = stats.pearsonr(plot_data_piv["Angular_model"], plot_data_piv["Angular_Humans"])
        ang_error.append(ang_r)

    humans_model = pd.DataFrame({'subj1':subj1, 'subj2':[model]*len(subj1),
                                    'Euclidean Error': euc_error, 'Angular Error': ang_error})
    humans_model['corr_rel'] = 'Humans-{}'.format(model)
    humans_models = humans_models.append(humans_model, ignore_index=True)


all_corr = pd.concat([humans_humans, humans_models])
plot_data = all_corr[['Euclidean Error','Angular Error', 'corr_rel']]
plot_data = plot_data.melt(id_vars=['corr_rel'])
sns_setup_small(sns, (8,6))
ax = sns.barplot(data=plot_data, x='value', y='corr_rel', hue='variable')
ax.set(xlabel='Correlation',ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(frameon=False)
ax.figure.savefig("figures/intact_gt_gazedperson_allcorr.png", dpi=300, bbox_inches='tight')
plt.close()

