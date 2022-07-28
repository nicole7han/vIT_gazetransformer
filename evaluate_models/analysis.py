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


''' human, cnn, transformer performance on intact, floating heads, headless bodies (transformer only trained with heads) '''
plot_data = results.copy()
plot_data = plot_data[plot_data['train_cond']=='Head']
plot_data['model'] = plot_data['model'].cat.rename_categories(['Humans', 'CNN', 'Head Gazetransformer'])
aov = pg.anova(dv='Euclidean_error', between=['model', 'test_cond'], data=plot_data,
             detailed=True)
print(aov)
postdoc =plot_data.pairwise_ttests(dv='Euclidean_error',
                                   between=['model', 'test_cond'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = postdoc[postdoc['p-corr']<0.05]
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
             (('Head Gazetransformer','floating heads'),('Head Gazetransformer','intact')), (('Head Gazetransformer','floating heads'),('Head Gazetransformer','headless bodies'))]
ps = [0.001, 0.001,0.001, 0.001,0.001,0.001]

sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond')
ax.set(xlabel='', ylabel='Euclidean Error')#, title='Transformer Trained with Heads')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')

add_stat_annotation(ax, data=plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside', verbose=2)
ax.legend(title='test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/modelxtest_cond.png", dpi=300, bbox_inches='tight')
plt.close()



''' human vs. CNN vs. Transformer correlation'''
plot_data = results.copy()
error = 'Angular' #Euclidean, Angular
train_cond = 'Body' #HeadBody, Head, Body

plot_data = plot_data[(plot_data['train_cond']==train_cond) & (plot_data['test_cond']=='intact')]
plot_data['model'] = plot_data['model'].cat.rename_categories(['Humans', 'CNN', '{} Gazetransformer'.format(train_cond)])
# plot_data_melt = pd.melt(plot_data, id_vars=['image','train_cond','test_cond','model'], value_vars='Euclidean_error')
plot_data_piv = plot_data.pivot(index="image", columns=["model"]).dropna().reset_index()
plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
plot_data_piv.columns = ['image','test_cond','test_cond','test_cond',
                         'Euclidean_Humans','Euclidean_CNN','Euclidean_Transformer',
                         'Angular_Humans','Angular_CNN','Angular_Transformer',
                         'train_cond','train_cond','train_cond']


cnn = plot_data_piv[['image', '{}_Humans'.format(error),'{}_CNN'.format(error)]]
cnn.columns = ['image', 'Humans', 'Model']
cnn['model'] = 'Human-CNN'
tran = plot_data_piv[['image', '{}_Humans'.format(error),'{}_Transformer'.format(error)]]
tran.columns = ['image', 'Humans', 'Model']
tran['model'] = 'Human-Transformer'
plotdata1 = pd.concat([cnn, tran])

labels = []
for cond in ['Human-CNN','Human-Transformer']:
    cond_data = plotdata1[plotdata1['model']==cond]
    r, p = stats.pearsonr(cond_data["Humans"], cond_data["Model"])
    r = round(r,2)
    ptxt = 'p<0.001' if p<0.001 else 'p={}'.format(round(p,3))
    label = 'r={}, {}'.format(r, ptxt)
    labels.append(label)

if error == 'Euclidean':
    sns_setup_small(sns)
    colrs = [custom_colors[2], custom_colors[4]]
    sns.set_palette(colrs)
    ax = sns.lmplot(x="Humans", y='Model', data=plotdata1,hue='model')
    ax.set(xlabel='Human Euclidean Error', ylabel='Model Euclidean Error')
    ax.ax.text(0.2, .50, labels[0], color=colrs[0],fontsize=20)
    ax.ax.text(0.2, .46, labels[1], color=colrs[1],fontsize=20)
    ax._legend.set_title('')
    ax.ax.spines['top'].set_color('white')
    ax.ax.spines['right'].set_color('white')
    ax.fig.savefig("figures/Euclidean_{}_intact.png".format(train_cond), dpi=300, bbox_inches='tight')
    plt.close()

if error == 'Angular':
    sns_setup_small(sns)
    colrs = [custom_colors[2], custom_colors[4]]
    sns.set_palette(colrs)
    ax = sns.lmplot(x="Humans", y='Model', data=plotdata1,hue='model')
    ax.set(xlabel='Human Angular Error (˚)', ylabel='Model Angular Error (˚)')
    ax.ax.text(20, 150, labels[0], color=colrs[0],fontsize=20)
    ax.ax.text(20, 135, labels[1], color=colrs[1],fontsize=20)
    ax._legend.set_title('')
    ax.ax.spines['top'].set_color('white')
    ax.ax.spines['right'].set_color('white')
    ax.fig.savefig("figures/Angular_{}_intact.png".format(train_cond), dpi=300, bbox_inches='tight')
    plt.close()





''' human vs. Transformer across different training conditions'''
plot_data = results.copy()
error = 'Euclidean' #Euclidean, Angular
img_cond = 'intact'
plot_data = plot_data[(plot_data['test_cond']==img_cond)]

plot_data['model'] = plot_data['model'].cat.rename_categories(['Humans', 'CNN', '{} Gazetransformer'.format(train_cond)])
plot_data_piv = plot_data.pivot(index=["image","train_cond"], columns=["model"]).dropna().reset_index()
plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
plot_data_piv.columns = ['image','train_cond','test_cond','test_cond','test_cond',
                         'Euclidean_Humans','Euclidean_CNN','Euclidean_Transformer',
                         'Angular_Humans','Angular_CNN','Angular_Transformer',
                         ]

plotdata1 = plot_data_piv[['image', 'train_cond','{}_Humans'.format(error),'{}_Transformer'.format(error)]]
plotdata1.columns = ['image','train_cond', 'Humans', 'Model']

labels = []
for cond in ['Body','Head','HeadBody']:
    cond_data = plotdata1[plotdata1['train_cond']==cond]
    r, p = stats.pearsonr(cond_data["Humans"], cond_data["Model"])
    r = round(r,2)
    ptxt = 'p<0.001' if p<0.001 else 'p={}'.format(round(p,3))
    label = 'r={}, {}'.format(r, ptxt)
    labels.append(label)


if error == 'Euclidean':
    sns_setup_small(sns)
    colrs = [custom_colors[2], custom_colors[4], custom_colors[6]]
    sns.set_palette(colrs)
    ax = sns.lmplot(x="Humans", y='Model', data=plotdata1,hue='train_cond')
    ax.set(xlabel='Human Euclidean Error', ylabel='Model Euclidean Error')
    ax.ax.text(0.38, 0.4, labels[0], color=colrs[0],fontsize=15)
    ax.ax.text(0.38, 0.36, labels[1], color=colrs[1],fontsize=15)
    ax.ax.text(0.38, 0.32, labels[2], color=colrs[2],fontsize=15)
    ax._legend.set_title('')
    ax.ax.spines['top'].set_color('white')
    ax.ax.spines['right'].set_color('white')
    ax.fig.savefig("figures/corr_Human_Transformers_EuclideanError_{}.png".format(train_cond,img_cond), dpi=300, bbox_inches='tight')
    plt.close()

if error == 'Angular':
    sns_setup_small(sns)
    colrs = [custom_colors[2], custom_colors[4], custom_colors[6]]
    sns.set_palette(colrs)
    ax = sns.lmplot(x="Humans", y='Model', data=plotdata1,hue='train_cond')
    ax.set(xlabel='Human Angular Error (˚)', ylabel='Model Angular Error (˚)')
    ax.ax.text(130, 150, labels[0], color=colrs[0],fontsize=15)
    ax.ax.text(130, 140, labels[1], color=colrs[1],fontsize=15)
    ax.ax.text(130, 130, labels[2], color=colrs[2],fontsize=15)
    ax._legend.set_title('')
    ax.ax.spines['top'].set_color('white')
    ax.ax.spines['right'].set_color('white')
    ax.fig.savefig("figures/corr_Human_Transformers_AngularError_{}.png".format(train_cond,img_cond), dpi=300, bbox_inches='tight')
    plt.close()




''' human vs. CNN'''
plot_data = results.copy()
error = 'Angular' #Euclidean, Angular
img_cond = 'intact'
plot_data = plot_data[(plot_data['test_cond']==img_cond)]

plot_data['model'] = plot_data['model'].cat.rename_categories(['Humans', 'CNN', '{} Gazetransformer'.format(train_cond)])
plot_data_piv = plot_data.pivot(index=["image","train_cond"], columns=["model"]).dropna().reset_index()
plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
plot_data_piv.columns = ['image','train_cond','test_cond','test_cond','test_cond',
                         'Euclidean_Humans','Euclidean_CNN','Euclidean_Transformer',
                         'Angular_Humans','Angular_CNN','Angular_Transformer',
                         ]

plotdata1 = plot_data_piv[['image', 'train_cond','{}_Humans'.format(error),'{}_CNN'.format(error)]]
plotdata1.columns = ['image','train_cond', 'Humans', 'Model']
plotdata1 = plotdata1.drop('train_cond',axis=1).drop_duplicates()

r, p = stats.pearsonr(plotdata1["Humans"], plotdata1["Model"])
r = round(r, 2)
ptxt = 'p<0.001' if p < 0.001 else 'p={}'.format(round(p, 3))
label = 'r={}, {}'.format(r, ptxt)

if error == 'Euclidean':
    sns_setup_small(sns)
    colrs = [custom_colors[2]]
    sns.set_palette(colrs)
    ax = sns.lmplot(x="Humans", y='Model', data=plotdata1)
    ax.set(xlabel='Human Euclidean Error', ylabel='Model Euclidean Error')
    ax.ax.text(0.38, 0.4, label, color=colrs[0],fontsize=15)
    ax.ax.spines['top'].set_color('white')
    ax.ax.spines['right'].set_color('white')
    ax.fig.savefig("figures/corr_Human_CNN_EuclideanError_{}.png".format(train_cond,img_cond), dpi=300, bbox_inches='tight')
    plt.close()

if error == 'Angular':
    sns_setup_small(sns)
    colrs = [custom_colors[2]]
    sns.set_palette(colrs)
    ax = sns.lmplot(x="Humans", y='Model', data=plotdata1)
    ax.set(xlabel='Human Angular Error (˚)', ylabel='Model Angular Error (˚)')
    ax.ax.text(130, 150, label, color=colrs[0],fontsize=15)
    ax.ax.spines['top'].set_color('white')
    ax.ax.spines['right'].set_color('white')
    ax.fig.savefig("figures/corr_Human_CNN_AngularError_{}.png".format(train_cond,img_cond), dpi=300, bbox_inches='tight')
    plt.close()



''' human vs. human '''
human_path = '/Users/nicolehan/Documents/Research/GazeExperiment/Mechanical turk/Analysis_absent'
results = glob.glob('{}/human*.xlsx'.format(human_path))
humans = pd.DataFrame()
for f in results:
    df = pd.read_excel(f)
    df.columns = ['human_est_x', 'human_est_y', 'subj', 'condition', 'movie', 'image']
    if 'intact' in f: Test_cond = 'intact'
    elif 'floating heads' in f: Test_cond = 'floating heads'
    elif 'headless bodies' in f: Test_cond = 'headless bodies'
    df = df.drop(['condition','movie'],axis=1)
    df = df.merge(image_info, on=['image'])
    df['Euclidean_error'] = np.sqrt(
        (df['gazed_x'] - df['human_est_x']) ** 2 + (df['gazed_y'] - df['human_est_y']) ** 2)
    df['Angular_error'] = df.apply(lambda r: compute_angle(r,'human'),axis=1)
    # df = df.groupby('image').mean().reset_index()  # mean subject error
    df['test_cond'] = Test_cond
    humans = pd.concat([humans,df])
humans = humans[(humans['subj']!=99401) & (humans['subj']!=99807)]
humans.to_excel('data/Human_estimations.xlsx',index=None)

humans = pd.read_excel('data/Human_estimations.xlsx')
humans = humans[humans['test_cond']=='intact']
subjects = list(np.unique(humans['subj']))
subj1, subj2, euc_error, ang_error = [], [], [], []
for s1 in subjects:
    print(s1)
    rest_subjects = subjects.copy()
    rest_subjects.remove(s1)
    for s2 in rest_subjects:
        # subjs = random.sample(list(subjects), 2)
        tempdata = humans[(humans['subj']==s1) | (humans['subj']==s2)]
        tempdata = tempdata[['image','subj','Euclidean_error','Angular_error']]
        tempdata= tempdata.pivot(index=["image"], columns=["subj"]).dropna().reset_index()
        tempdata.columns = tempdata.columns.droplevel(1)
        tempdata.columns = ['image', 'Euclidean_subj1', 'Euclidean_subj2', 'Angular_subj1', 'Angular_subj2']
        subj1.append(s1)
        subj2.append(s2)
        r, p = stats.pearsonr(tempdata["Euclidean_subj1"], tempdata["Euclidean_subj2"])
        euc_error.append(r)
        r, p = stats.pearsonr(tempdata["Angular_subj1"], tempdata["Angular_subj2"])
        ang_error.append(r)

human_corr = pd.DataFrame({'subj1':subj1, 'subj2':subj2, 'Euclidean Error':euc_error, 'Angular Error':ang_error})
human_corr.to_excel('data/Human_error_corr.xlsx',index=None)
print(human_corr.mean())
r1,p1 = stats.ttest_ind(human_corr['Euclidean Error'],[0]*len(human_corr))
r2,p2 = stats.ttest_ind(human_corr['Angular Error'],[0]*len(human_corr))

plot_data = human_corr.melt(id_vars=['subj1','subj2'])
sns_setup_small(sns, (6,6))
ax = sns.barplot(data=plot_data, x='variable', y='value',color=setpallet[0])
change_width(ax, .4)
ax.set(xlabel='',ylabel='Mean Correlation')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig("figures/corr_Human_Human_{}.png".format(img_cond), dpi=300, bbox_inches='tight')
plt.close()




''' Human-Human, Human-CNN, Human-3transformers Correlation '''
# for both euclidean and angular error
# human-human correlation
humans = pd.read_excel('data/Human_error_corr.xlsx')
humans['corr_rel'] = 'Human-Human'

# human-CNN correlation
humans_CNN = results[(results['model']=='CNN') | (results['model']=='Humans')]
humans_CNN = humans_CNN[(humans_CNN['test_cond']=='intact') & (humans_CNN['train_cond']=='Head')]
humans_CNN = humans_CNN.drop('test_cond',axis=1)
plot_data_piv = humans_CNN.pivot(index=["image","train_cond"], columns=["model"]).dropna().reset_index()
plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
plot_data_piv.columns = ['image', 'train_cond', 'Euclidean_CNN', 'Euclidean_Humans',
       'Angular_CNN', 'Angular_Humans']
eu_r, eu_p = stats.pearsonr(plot_data_piv["Euclidean_CNN"], plot_data_piv["Euclidean_Humans"])
eu_r = round(eu_r, 2)
eu_p = round(eu_p, 5)
ang_r, ang_p = stats.pearsonr(plot_data_piv["Angular_CNN"], plot_data_piv["Angular_Humans"])
ang_r = round(ang_r, 2)
ang_p = round(ang_p, 5)
humans_CNN_corr = pd.DataFrame()
humans_CNN_corr = humans_CNN_corr.append({'subj1':'Humans', 'subj2':'CNN',
                                'Euclidean Error': eu_r, 'Euclidean Error p':eu_p,
                                'Angular Error': ang_r, 'Angular Error p':ang_p}, ignore_index=True)
humans_CNN_corr['corr_rel'] = 'Humans-CNN'

# human-transformer correlation
humans_transformer = results[(results['model']=='Transformer') | (results['model']=='Humans')]
humans_transformer = humans_transformer[humans_transformer['test_cond']=='intact']
humans_transformer = humans_transformer.drop('test_cond',axis=1)
plot_data_piv = humans_transformer.pivot(index=["image","train_cond"], columns=["model"]).dropna().reset_index()
plot_data_piv.columns = plot_data_piv.columns.droplevel(1)
plot_data_piv.columns = ['image', 'train_cond', 'Euclidean_Transformer', 'Euclidean_Humans',
       'Angular_Transformer', 'Angular_Humans']
humans_trans_corr = pd.DataFrame()
for cond in ['HeadBody', 'Head','Body']:
    tempdata = plot_data_piv[plot_data_piv['train_cond']==cond]
    eu_r, eu_p = stats.pearsonr(tempdata["Euclidean_Transformer"], tempdata["Euclidean_Humans"])
    eu_r = round(eu_r, 2)
    eu_p = round(eu_p, 5)
    ang_r, ang_p = stats.pearsonr(tempdata["Angular_Transformer"], tempdata["Angular_Humans"])
    ang_r = round(ang_r, 2)
    ang_p = round(ang_p, 5)
    humans_trans_corr = humans_trans_corr.append({'subj1':'Humans', 'subj2':'{} Transformer'.format(cond),
                                                  'Euclidean Error': eu_r, 'Euclidean Error p':eu_p,
                                                  'Angular Error': ang_r, 'Angular Error p':ang_p,
                                                  'corr_rel': 'Human-{}Transformer'.format(cond)},
                                                 ignore_index=True)

all_corr = pd.concat([humans, humans_CNN_corr, humans_trans_corr])
plot_data = all_corr[['Euclidean Error','Angular Error', 'corr_rel']]
plot_data = plot_data.melt(id_vars=['corr_rel'])
sns_setup_small(sns, (8,6))
ax = sns.barplot(data=plot_data, x='value', y='corr_rel', hue='variable')
ax.set(xlabel='Correlation',ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(frameon=False)
ax.figure.savefig("figures/allcorr_intact.png", dpi=300, bbox_inches='tight')
plt.close()



''' Human, CNN, 3 Transformers performance on intact image '''
plot_data = results.copy()
plot_data = plot_data[plot_data['test_cond']=='intact']
humans = plot_data[(plot_data['model']=='Humans') & (plot_data['train_cond']=='Head')]
CNN = plot_data[(plot_data['model']=='CNN') & (plot_data['train_cond']=='Head')]
transformers = plot_data[(plot_data['model']=='Transformer')]
def label_transformer(row):
    return row['train_cond'] + ' Transformer'
transformers['model'] = transformers.apply(label_transformer, axis=1)
plot_data = pd.concat([humans,CNN,transformers])
plot_data['model'] = plot_data['model'].astype('category')
plot_data['model'] = plot_data['model'].cat.rename_categories(['Humans', 'CNN', 'HeadBody Gazetransformer',
                                                               'Head Gazetransformer','Body Gazetransformer',])
plot_data = plot_data[['Euclidean_error','Angular_error','model']]
# plot_data = plot_data.melt(id_vars='model')
# aov = pg.anova(dv='Euclidean_error', between=['model'], data=plot_data,
#              detailed=True)
# print(aov)
# postdoc =plot_data.pairwise_ttests(dv='Euclidean_error',
#                                    between=['model'],
#                                    padjust='fdr_bh',
#                                    parametric=True).round(3)
# sig_results = postdoc[postdoc['p-corr']<0.05]
# box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
#              (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
#              (('Head Gazetransformer','floating heads'),('Head Gazetransformer','intact')), (('Head Gazetransformer','floating heads'),('Head Gazetransformer','headless bodies'))]
# ps = [0.001, 0.001,0.001, 0.001,0.001,0.001]

sns_setup_small(sns, (8,6))
error = 'Euclidean_error'
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[0])
ax.set(xlabel='Euclidean Error', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
# add_stat_annotation(ax, data=plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond',
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside', verbose=2)
# ax.legend(title='test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/Euclidean_human_allmodels_intact.png", dpi=300, bbox_inches='tight')
plt.close()

sns_setup_small(sns, (8,6))
error = 'Angular_error'
ax = sns.barplot(data = plot_data, x = error, y = 'model',color=setpallet[1])
ax.set(xlabel='Angular Error', ylabel='')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
# add_stat_annotation(ax, data=plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond',
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside', verbose=2)
# ax.legend(title='test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/Angular_human_allmodels_intact.png", dpi=300, bbox_inches='tight')
plt.close()


