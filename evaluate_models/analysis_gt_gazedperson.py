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
custom_colors = sns.color_palette("Set1", 10)
basepath = '/Users/nicolehan/Documents/Research/gazetransformer'





''' PART I Human, CNN, 3 Transformer performance (gazed person as groundtruth) '''
summaries = glob.glob('data/GroundTruth_gazedperson/*summary*')
results = pd.DataFrame()
for f in summaries:
    data = pd.read_excel(f)
    results = results.append(data)
results['test_cond'] = results['test_cond'].astype('category')
results['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
results['model'] = results['model'].astype('category')
results['model'].cat.reorder_categories(['Humans', 'CNN', 'HeadBody Transformer', 'Head Transformer', 'Body Transformer'], inplace=True)

plot_data = results[results['test_cond']=='intact']
plot_data = plot_data[[ 'Euclidean_error', 'Angular_error','model']]

error = 'Angular_error'
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

# sns_setup_small(sns, (8,6))
# ax = sns.barplot(data = plot_data, x = 'model', y = error ,color=setpallet[0])
# ax.set(xlabel='', ylabel='Euclidean Error')
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# add_stat_annotation(ax, data=plot_data, x='model', y=error,
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside', verbose=2)
# ax.legend(title='', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
# plt.xticks(rotation=90)
# ax.figure.savefig("figures/intact_gt_gazedperson_{}2.png".format(error), dpi=300, bbox_inches='tight')
# plt.close()



sns_setup_small(sns, (8,6))
error = 'Angular_error'
ax = sns.barplot(data = plot_data, x = 'model', y = error, color=setpallet[1])
ax.set(xlabel='', ylabel='Angular Error (˚)')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
add_stat_annotation(ax, data=plot_data, x='model', y=error,
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
plt.xticks(rotation=90)
ax.figure.savefig("figures/intact_gt_gazedperson_{}2.png".format(error), dpi=300, bbox_inches='tight')
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