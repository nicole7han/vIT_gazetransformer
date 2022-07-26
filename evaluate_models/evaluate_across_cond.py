import pandas as pd
import glob
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'


Trained_cond = 'HeadBody'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)

'''transformer results'''
transformer = pd.DataFrame()
for epoch in [100,300,120]:
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'

        df = df.groupby('image').mean().reset_index()  # compute estimation for each image
        df['test_cond'] = Test_cond
        transformer = pd.concat([transformer,df])

image_info = transformer[['image','gazed_x','gazed_y']].drop_duplicates()
transformer['Euclidean_error'] = np.sqrt( (transformer['gazed_x']-transformer['transformer_est_x'])**2 + (transformer['gazed_y']-transformer['transformer_est_y'])**2 )
transformer = transformer.groupby(['image','test_cond']).mean().reset_index()
transformer = transformer[['test_cond','Euclidean_error']]
transformer['model'] = 'Transformer'

'''CNN results'''
results = glob.glob('{}/chong*.csv'.format(basepath))
cnn = pd.DataFrame()
for f in results:
    df = pd.read_csv(f)
    if 'intact' in f: Test_cond = 'intact'
    elif 'nb' in f: Test_cond = 'floating heads'
    elif 'nh' in f: Test_cond = 'headless bodies'

    df = df.groupby('image').mean().reset_index()  # compute estimation for each image
    df['test_cond'] = Test_cond
    cnn = pd.concat([cnn,df])

cnn = cnn.merge(image_info, on=['image'])
cnn['Euclidean_error'] = np.sqrt( (cnn['gazed_x']-cnn['chong_est_x'])**2 + (cnn['gazed_y']-cnn['chong_est_y'])**2 )
cnn = cnn[['test_cond','Euclidean_error']]
cnn['model'] = 'CNN'


'''human results'''
human_path = '/Users/nicolehan/Documents/Research/GazeExperiment/Mechanical turk/Analysis_absent'
results = glob.glob('{}/human*.xlsx'.format(human_path))
humans = pd.DataFrame()
for f in results:
    df = pd.read_excel(f)
    if 'intact' in f: Test_cond = 'intact'
    elif 'floating heads' in f: Test_cond = 'floating heads'
    elif 'headless bodies' in f: Test_cond = 'headless bodies'
    df = df.drop(['condition','movie'],axis=1)
    df = df.merge(image_info, on=['image'])
    df['Euclidean_error'] = np.sqrt(
        (df['gazed_x'] - df['human_x']) ** 2 + (df['gazed_y'] - df['human_y']) ** 2)
    df = df.groupby('image').mean().reset_index()  # mean subject error
    df['test_cond'] = Test_cond
    humans = pd.concat([humans,df])

humans = humans[['test_cond','Euclidean_error']]
humans['model'] = 'Humans'


plot_data = pd.concat([transformer, cnn, humans])
plot_data['test_cond'] = plot_data['test_cond'].astype('category')
plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
plot_data.to_excel('data/{}_summary.xlsx'.format(Trained_cond), index=None)
aov = pg.anova(dv='Euclidean_error', between=['model', 'test_cond'], data=plot_data,
             detailed=True)
print(aov)
postdoc =plot_data.pairwise_ttests(dv='Euclidean_error',
                                   between=['model', 'test_cond'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = postdoc[postdoc['p-corr']<0.05]

from statannot import add_stat_annotation
sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond')
ax.set(xlabel='', ylabel='Euclidean Error', title='Transformer Trained: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
# box_pairs_model = [('cnn','transformer'),('humans','transformer')]
# ps_models = [0.001, 0.001]
# add_stat_annotation(ax, data=plot_data, x = 'model', y = 'Euclidean_error',
#                     box_pairs= box_pairs_model, perform_stat_test=False, pvalues=ps_models,
#                     loc='outside',line_offset=0.015, line_offset_to_box=0.005, verbose=2)
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
             (('Transformer','intact'),('Transformer','headless bodies'))]
ps = [0.001, 0.001,0.001, 0.001,0.04]
add_stat_annotation(ax, data=plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='inside',line_offset=0.1, line_offset_to_box=0.005, verbose=0)
ax.legend(title='Test condition', loc='upper left', frameon=False)
ax.figure.savefig("figures/{}_model_comparison.jpg".format(Trained_cond), bbox_inches='tight')
plt.close()


