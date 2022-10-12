import pandas as pd
import glob, os, math
from script.model import *
import statsmodels.api as sm
from itertools import combinations
from statsmodels.stats.multitest import multipletests as mt
from statsmodels.formula.api import ols
import pingouin as pg
from scipy import stats
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
from evaluate_models.utils import *
setpallet = sns.color_palette("Set2")
bluepallet = sns.color_palette("Blues_r")
custom_colors = sns.color_palette("Set1", 10)
basepath = '/Users/nicolehan/Documents/Research/gazetransformer'





''' PART I Human, CNN, 3 Transformer performance (gazed person as groundtruth) '''
baselines = glob.glob('data/GroundTruth_gazedperson/*Perm*') # permutation baselines
baseline = pd.DataFrame()
for f in baselines:
    data = pd.read_excel(f)
    baseline = baseline.append(data)
baseline['test_cond'] = baseline['test_cond'].astype('category')
baseline['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
baseline['model'] = baseline.apply(lambda r: r['model'].split(' ')[1] if 'Head CNN' in r['model'] else r['model'], axis=1)
baseline['model'] = baseline.apply(lambda r: r['model'].replace(' ','\n') if 'Transformer' in r['model'] else r['model'], axis=1)
baseline['model'] = baseline['model'].astype('category')
#baseline['model'].cat.reorder_categories(['Humans', 'Head CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer'], inplace=True)
baseline['model'].cat.reorder_categories(['Humans', 'CNN', 'HeadBody\nTransformer', 'Head\nTransformer', 'Body\nTransformer'], inplace=True)
baseline_quantile = baseline.groupby(['test_cond','model']).quantile([.025,0.975]).reset_index()
baseline = baseline.groupby(['test_cond','model']).mean().reset_index()

summaries = glob.glob('data/GroundTruth_gazedperson/*summary*')
results = pd.DataFrame()
for f in summaries:
    data = pd.read_excel(f)
    results = results.append(data)
results['test_cond'] = results['test_cond'].astype('category')
results['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
results['model'] = results.apply(lambda r: r['model'].split(' ')[1] if 'Head CNN' in r['model'] else r['model'], axis=1)
results['model'] = results.apply(lambda r: r['model'].replace(' ','\n') if 'Transformer' in r['model'] else r['model'], axis=1)
results['model'] = results['model'].astype('category')
#results['model'].cat.reorder_categories(['Humans', 'CNN', 'Transformer'], inplace=True)
#models = ['Humans', 'CNN', 'Transformer']
results['model'].cat.reorder_categories(['Humans', 'CNN', 'HeadBody\nTransformer', 'Head\nTransformer', 'Body\nTransformer'], inplace=True)
models = ['Humans', 'CNN', 'HeadBody\nTransformer', 'Head\nTransformer', 'Body\nTransformer']

# plot_data = results[results['test_cond']=='intact']
plot_data = results.copy()
plot_data = plot_data[['test_cond', 'Euclidean_error', 'Angular_error','model']]

error = 'Euclidean_error'
aov_data = plot_data[[error, 'test_cond', 'model']].melt(id_vars=['test_cond','model'])
aov = pg.anova(dv='value', between=['test_cond','model'], data=aov_data,
             detailed=True)
print(aov)
posthoc = aov_data.pairwise_ttests(dv='value',
                                   between=['model','test_cond'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = posthoc[posthoc['p-corr']<0.05]
sig_results = sig_results[sig_results['Contrast']=='model * test_cond']
box_pairs = []
ps = []
for _, row in sig_results.iterrows():
    box_pairs.append(((row['model'],row['A']),(row['model'],row['B'])))
    ps.append(max(0.001, row['p-corr']))
    
    
posthoc = aov_data.pairwise_ttests(dv='value',
                                   between=['model'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = posthoc[posthoc['p-corr']<0.05]
sig_results = sig_results[sig_results['Contrast']=='model']
box_pairs1 = []
ps1 = []
for _, row in sig_results.iterrows():
    box_pairs1.append((row['A'],row['B']))
    ps1.append(max(0.001, row['p-corr']))
# test angular error (paird images) significance between condition: intact, Headbody transformer vs. Head transformer
from bioinfokit.analys import stat
res = stat()
res.tukey_hsd(df=aov_data, res_var='value', xfac_var=['model','test_cond'], anova_model='value~C(model)+C(test_cond)+C(model):C(test_cond)')
res.tukey_summary



 # plot euclidean error
 sns_setup_small(sns, (12,9))
 error = 'Euclidean_error'
 ax = sns.barplot(data = plot_data, x = 'model', y = error , hue='test_cond', palette=bluepallet)
 ax.set(xlabel='', ylabel='Euclidean Error', ylim=[0,0.85])
 ax.spines['top'].set_color('white')
 ax.spines['right'].set_color('white')
 add_stat_annotation(ax, data=plot_data, x='model', y=error, hue='test_cond',
                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                     loc='outside', verbose=2)
# add_stat_annotation(ax, data=plot_data, x='model', y=error,
#                     box_pairs= box_pairs1, perform_stat_test=False, pvalues=ps1,
#                     loc='inside', verbose=2)
# add_stat_annotation(ax, data=plot_data, x='model', y=error,
#                     box_pairs= [('Body\nTransformer', 'Humans')], perform_stat_test=False, pvalues=[0.06],
#                     loc='inside',  text_format='full', verbose=2)
 ax.legend(title='', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
 

ps1.append(0.06)
# plt.xticks(rotation=90)
 xcen = 0.1
 shade_width = 0.02
 cen_gap = 0.2
 bar_width = 0.16
 for model in models:
     m = baseline[baseline['model']==model]
     q1 = baseline_quantile[(baseline_quantile['model']==model) & (baseline_quantile['level_2']==.025)]
     q2 = baseline_quantile[(baseline_quantile['model'] == model) & (baseline_quantile['level_2'] == .975)]
     for i, cond in enumerate(['intact','floating heads','headless bodies']):
#         left = xcen + (0.16 / 3) * (i - 1) - 0.02
#         right = xcen + (0.16 / 3) * (i - 1) + 0.02
         left = xcen + (bar_width/ 3) * (i - 1) - shade_width
         right = xcen + (bar_width / 3) * (i - 1) + shade_width
         print(left, right)
         x, a, b  = m[m['test_cond']==cond].Euclidean_error.item(), \
                    q1[q1['test_cond'] == cond].Euclidean_error.item(), \
                    q2[q2['test_cond'] == cond].Euclidean_error.item(),
         plt.axhspan(ymin=a, ymax=b,
                     xmin=left,
                     xmax=right,
                     facecolor='0.5', alpha=0.2)
         plt.axhline(y=x,
                     xmin=left,
                     xmax=right,
                     color='0.8')
     xcen += cen_gap
 ax.figure.savefig("figures/gt_gazedperson_{}_allcond.png".format(error), dpi=300, bbox_inches='tight')
 plt.close()


# plot angular error
sns_setup_small(sns, (8,6))
error = 'Angular_error'
ax = sns.barplot(data = plot_data, x = 'model', y = error, hue='test_cond',palette=bluepallet)
ax.set(xlabel='', ylabel='Angular Error (˚)')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
add_stat_annotation(ax, data=plot_data, x='model', y=error, hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
ax.legend(title='', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
plt.xticks(rotation=90)
xcen = 0.1
for model in ['Humans', 'Head CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']:
    m = baseline[baseline['model']==model]
    q1 = baseline_quantile[(baseline_quantile['model']==model) & (baseline_quantile['level_2']==.025)]
    q2 = baseline_quantile[(baseline_quantile['model'] == model) & (baseline_quantile['level_2'] == .975)]
    for i, cond in enumerate(['intact','floating heads','headless bodies']):
        x, a, b  = m[m['test_cond']==cond].Angular_error.item(), \
                   q1[q1['test_cond'] == cond].Angular_error.item(), \
                   q2[q2['test_cond'] == cond].Angular_error.item(),
        plt.axhspan(ymin=a, ymax=b,
                    xmin=xcen + (0.16 / 3) * (i - 1) - 0.02,
                    xmax=xcen + (0.16 / 3) * (i - 1) + 0.02,
                    facecolor='0.5', alpha=0.2)
        plt.axhline(y=x,
                    xmin=xcen + (0.16 / 3) * (i - 1) - 0.02,
                    xmax=xcen + (0.16 / 3) * (i - 1) + 0.02,
                    color='0.8')
    xcen +=.2
ax.figure.savefig("figures/gt_gazedperson_{}_allcond.png".format(error), dpi=300, bbox_inches='tight')
plt.close()



''' PART II Human-Human, Human-CNN, Human-Transformer Error Correlation on Euclidean and Angular Error '''
# calculate human-human error wrt gazed person
# image_info = pd.read_excel('data/GroundTruth_gazedperson/image_info.xlsx')
# human_path = '/Users/nicolehan/Documents/Research/GazeExperiment/Mechanical turk/Analysis_absent'
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
humans_humans['corr_rel'] = 'Humans-Humans'


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


# boostrapping
boot_data = all_corr
boot_data = boot_data.groupby(['corr_rel','subj1']).mean().reset_index()
cond = list(np.unique(boot_data.corr_rel))
conditions = set(list(combinations(cond, 2)))
pvals = pd.DataFrame()
for var in ['Euclidean Error','Angular Error']:
    print(var)
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
    box_pairs.append(((cond1, var), (cond2,var)))


sns_setup_small(sns, (8,6))
ax = sns.barplot(data=plot_data, x='corr_rel', y='value', hue='variable')
ax.set(xlabel='',ylabel='Correlation')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(frameon=False)
add_stat_annotation(ax, data=plot_data, x='corr_rel', y='value', hue='variable',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
ax.legend(title='', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
plt.xticks(rotation=90, fontsize=20)
ax.figure.savefig("figures/intact_gt_gazedperson_allcorr_sig.png", dpi=300, bbox_inches='tight')
plt.close()






''' PART III Human-Human, Human-CNN, Human-Transformer Vector Angle Correlation '''
files = glob.glob('data/GroundTruth_gazedperson/*vectors.xlsx')
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
# human_corr = pd.DataFrame({'subj1':subj1, 'subj2':subj2, 'vec_angle_corr':corr})
# human_corr['corr_rel'] = 'Humans-Humans'
# human_corr.to_excel('data/GroundTruth_gazedperson/Human_intact_vec_angle_corr.xlsx',index=None)
humans_humans = pd.read_excel('data/GroundTruth_gazedperson/Human_intact_vec_angle_corr.xlsx')

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
    box_pairs.append(((cond1, var), (cond2,var)))


sns_setup_small(sns, (8,6))
ax = sns.barplot(data=plot_data, x='corr_rel', y= 'value',color=setpallet[2])
ax.set(xlabel='',ylabel='Correlation',title='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
add_stat_annotation(ax, data=plot_data, x='corr_rel', y='value', hue='variable',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
plt.xticks(rotation=90, fontsize=20)
ax.figure.savefig("figures/intact_gt_gazedperson_vec_ang_corr.png", dpi=300, bbox_inches='tight')
plt.close()






''' PART IV Visualize Estimation Vectors '''
img_path = '/Users/nicolehan/Documents/Research/gazetransformer/gaze_video_data/transformer_all_img_intact'

data_path = 'data/GroundTruth_gazedperson'
files = glob.glob('{}/*vectors*'.format(data_path)) # get all vector information
results = pd.DataFrame()
for f in files:
    df = pd.read_excel(f)
    df.columns = [x if 'est' not in x else '_'.join(x.split('_')[1:]) for x in df.columns ]
    results = results.append(df, ignore_index=True)
results = results[results['cond']=='intact'].drop('cond',axis=1)
results = results.groupby(['image','model','gazer']).mean().reset_index()


# # plot individual images
# image_names = os.listdir(img_path)
# for img in image_names:
# # img = image_names[0]
#     img_data = results[(results['image'] == img)]
#     gazers = np.unique(img_data['gazer'])
#     for gazer in gazers:
#         plot_data = img_data[(img_data['gazer']==gazer)]
#         if len(plot_data) == 5:
#             image = np.array(Image.open('{}/{}'.format(img_path,img)))
#             plot_vectors(image, plot_data, gazer=gazer)
#             plt.savefig("Figures/vectors/{}_gazer{}.jpg".format(img,gazer), bbox_inches='tight')
#             plt.close()


# plot all images registered to groundtruth vector
image_names = os.listdir(img_path)
plot_data = results[['image','model','gazer','gaze_start_x','gaze_start_y','gazed_x','gazed_y','est_x','est_y']]
# compute angular error (with signs, flip everything to the right direction)
# 1. positive means clockwise rotation, negative means clockwise rotation
plot_data['vec_ang2hor'] = -plot_data.apply(compute_ang2hori, axis=1) #groundtruth vector angle relative to the horizontal direction
# 2. rotate estxy relative to gazer, in the opposite direction as groundtruth vector to cancel its effect
plot_data['estxy_rotate'] = plot_data.apply(lambda r: rotate(r, 'est', 'vec_ang2hor'),axis=1)  # calculate estimation xy after rotation
plot_data[['est_x1', 'est_y1']] = pd.DataFrame(plot_data['estxy_rotate'].tolist(), index=plot_data.index)
# 3. get final estimation xy location relative to horizontal vector (groundtruth vector)
plot_data['est_y1_2_gazer'] = plot_data['est_y1'] - plot_data['gaze_start_y']  # positive: below horizontal, negative: above horizontal
# 4. compute estimation vector relative to groundtruth vector
angle_errors_signed = []
for _, r in plot_data.iterrows():
    gaze_start_x, gaze_start_y, gazed_x, gazed_y = r['gaze_start_x'], r['gaze_start_y'],r['gazed_x'], r['gazed_y']
    v1 = np.array([gazed_x - gaze_start_x, gazed_y - gaze_start_y])
    unit_vector_1 = v1 / np.linalg.norm(v1)

    if r['est_y1_2_gazer'] > 0:
        sign = -1  # the relative gazer vector is below horizontal (human vector)
    else:
        sign = 1  # the relative gazer vector is above horizontal (human vector)

    v2 = np.array([r['est_x'] - gaze_start_x, r['est_y'] - gaze_start_y])
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi

    angle_errors_signed.append(sign * round(angle, 2))

plot_data['signed_angle2gtvec'] = angle_errors_signed
plot_data['ang_rad'] = plot_data['signed_angle2gtvec']*np.pi/180

# plot estimation vector
model_orders = ['Humans','CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']
colors = sns.color_palette("Set2")
colors = colors[:4] + [colors[6]]

for i, model in enumerate(model_orders):
    tempdata = plot_data[plot_data['model']==model]
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    circular_hist(ax, tempdata['ang_rad'], color=colors[i])
    ax.figure.savefig("figures/polar_{}_vector2gt.png".format(model), dpi=300, bbox_inches='tight')
    plt.close()




# plot all images registered to human mean vector
image_names = os.listdir(img_path)
plot_data = results[['image','model','gazer','gaze_start_x','gaze_start_y','est_x','est_y']]
model_vectors = plot_data[plot_data['model']!='Humans']
human_vectors = plot_data[plot_data['model']=='Humans']
human_vectors = human_vectors[['image','gazer','est_x','est_y']]
human_vectors.columns = ['image','gazer','gazed_x','gazed_y']
plot_data = model_vectors.merge(human_vectors, on=['image','gazer'])

# compute angular error (with signs, flip everything to the right direction)
# 1. negative means clockwise rotation, positive means clockwise rotation
# because y axis is flipped in our dataset, needs a negative sign to flip the angle such that positive clockwise, negative counterclockwise
plot_data['humanvec_ang2hor'] = -plot_data.apply(compute_ang2hori, axis=1)
# 2. rotate estxy relative to gazer, in the same direction as human vector relative to the horizontal direction
plot_data['estxy_rotate'] = plot_data.apply(lambda r: rotate(r, 'est'),axis=1)  # calculate estimation xy with the same rotation
plot_data[['est_x1', 'est_y1']] = pd.DataFrame(plot_data['estxy_rotate'].tolist(), index=plot_data.index)
# 3. get final estimation xy location relative to horizontal (human vector)
plot_data['est_y1_2_gazer'] = plot_data['est_y1'] - plot_data['gaze_start_y']  # positive: below horizontal, negative: above horizontal
# 4. compute estimation vector relative to human vector direction

angle_errors_signed = []
for _, r in plot_data.iterrows():
    gaze_start_x, gaze_start_y, gazed_x, gazed_y = r['gaze_start_x'], r['gaze_start_y'],r['gazed_x'], r['gazed_y']
    v1 = np.array([gazed_x - gaze_start_x, gazed_y - gaze_start_y])
    unit_vector_1 = v1 / np.linalg.norm(v1)

    if r['est_y1_2_gazer'] > 0:
        sign = -1  # the relative gazer vector is below horizontal (human vector)
    else:
        sign = 1  # the relative gazer vector is above horizontal (human vector)

    v2 = np.array([r['est_x'] - gaze_start_x, r['est_y'] - gaze_start_y])
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi

    angle_errors_signed.append(sign * round(angle, 2))

plot_data['signed_angle2humanvec'] = angle_errors_signed
plot_data['ang_rad'] = plot_data['signed_angle2humanvec']*np.pi/180

# plot estimation vector
model_orders = ['CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']
colors = sns.color_palette("Set2")
colors = colors[:4] + [colors[6]]

for i, model in enumerate(model_orders):
    tempdata = plot_data[plot_data['model']==model]
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    circular_hist(ax, tempdata['ang_rad'], color=colors[i])
    ax.figure.savefig("figures/polar_{}_vector2human.png".format(model), dpi=300, bbox_inches='tight')
    plt.close()



''' Part V. Visualize Model Raw Estimation (see the variance in model estimations) '''
import matplotlib.patches as  mpatches
data_path = 'data/GroundTruth_gazedperson'
files = glob.glob('{}/*vectors*'.format(data_path)) # get all vector information
results = pd.DataFrame()
for f in files:
    df = pd.read_excel(f)
    df.columns = [x if 'est' not in x else '_'.join(x.split('_')[1:]) for x in df.columns ]
    results = results.append(df, ignore_index=True)

for model in ['Humans', 'Head CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer']:
    plot_data = results[results['model'] == model]
    if model == 'Humans':
        plot_data = plot_data.groupby(['cond', 'image', 'subj']).mean().reset_index()  # take average across gazers for each subject
        plot_data = plot_data.groupby(['cond', 'image']).mean().reset_index() # take average across subjects
    else:
        plot_data = plot_data.groupby(['cond','image']).mean().reset_index() # take average across gazers
    plot_data = plot_data[['cond','image','est_x','est_y','gazed_x','gazed_y']]
    
    # gazed person xy density map
    gazed_info = plot_data[['gazed_x','gazed_y']].drop_duplicates()
    ax = sns.kdeplot(data=plot_data, x=gazed_info["gazed_x"], y=gazed_info["gazed_y"], fill=True, alpha=.8, cmap="Blues", label='gazed person')
    # estimation xy density map
    ax = sns.kdeplot(data=plot_data, x="est_x", y="est_y", fill=True, alpha=.5, cmap="Reds", label='estimation')
    handles = [mpatches.Patch(facecolor=plt.cm.Reds(100), label="estimation"),
           mpatches.Patch(facecolor=plt.cm.Blues(100), label="gazed peron")]
    plt.legend(handles=handles, frameon=False)
    ax.set(xlim=[0,1],ylim=[0.2,.8])
    ax.figure.savefig("figures/{}_raw_estimation2.png".format(model), dpi=300, bbox_inches='tight')
    plt.close()
    