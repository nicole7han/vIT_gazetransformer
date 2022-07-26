import pandas as pd
import glob
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")

def compute_angle(row, model):
    vector_1 = [row['gazed_x'] - row['gaze_start_x'], row['gazed_y'] - row['gaze_start_y']]
    vector_2 = [row['{}_est_x'.format(model)] - row['gaze_start_x'], row['{}_est_y'.format(model)] - row['gaze_start_y']]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi  # angle in degrees
    return angle

basepath = '/Users/nicolehan/Documents/Research/gazetransformer'

Trained_cond = 'Head'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)

'''transformer results'''
transformer = pd.DataFrame()
for epoch in [108,60,150]:
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'

        df = df.groupby('image').mean().reset_index()  # compute estimation for each image
        df['test_cond'] = Test_cond
        transformer = pd.concat([transformer,df])

image_info = transformer[['image','gaze_start_x','gaze_start_y','gazed_x','gazed_y']].drop_duplicates()
transformer['Euclidean_error'] = np.sqrt( (transformer['gazed_x']-transformer['transformer_est_x'])**2 + (transformer['gazed_y']-transformer['transformer_est_y'])**2 )
transformer['Angular_error'] = transformer.apply(lambda r: compute_angle(r,'transformer'),axis=1)
transformer = transformer.groupby(['image','test_cond']).mean().reset_index()
transformer = transformer[['test_cond','Euclidean_error','Angular_error']]
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

cnn = cnn.merge(image_info[['image', 'gazed_x', 'gazed_y']], on=['image'])
cnn['Euclidean_error'] = np.sqrt( (cnn['gazed_x']-cnn['chong_est_x'])**2 + (cnn['gazed_y']-cnn['chong_est_y'])**2 )
cnn['Angular_error'] = cnn.apply(lambda r: compute_angle(r,'chong'),axis=1)
cnn = cnn[['test_cond','Euclidean_error','Angular_error']]
cnn['model'] = 'CNN'


'''human results'''
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
    df = df.groupby('image').mean().reset_index()  # mean subject error
    df['test_cond'] = Test_cond
    humans = pd.concat([humans,df])

humans = humans[['test_cond','Euclidean_error','Angular_error']]
humans['model'] = 'Humans'


plot_data = pd.concat([transformer, cnn, humans])
plot_data['test_cond'] = plot_data['test_cond'].astype('category')
plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
plot_data.to_excel('data/{}_summary.xlsx'.format(Trained_cond), index=None)



plot_data = pd.read_excel('data/{}_summary.xlsx'.format(Trained_cond))
plot_data['test_cond'] = plot_data['test_cond'].astype('category')
plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
plot_data['model'] = plot_data['model'].astype('category')
plot_data['model'].cat.reorder_categories(['Humans', 'CNN', 'Transformer'], inplace=True)

'''Euclidean Error '''
error = 'Euclidean' # Angular or Euclidean
aov = pg.anova(dv='{}_error'.format(error), between=['model', 'test_cond'], data=plot_data,
             detailed=True)
print(aov)
postdoc =plot_data.pairwise_ttests(dv='{}_error'.format(error),
                                   between=['model', 'test_cond'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = postdoc[postdoc['p-corr']<0.05]

sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond')
ax.set(xlabel='', ylabel='{} Error'.format(error), title='Transformer Trained: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
             (('Transformer','floating heads'),('Transformer','intact')), (('Transformer','floating heads'),('Transformer','headless bodies'))]
ps = [0.001, 0.001,0.001, 0.001,0.001,0.01]
add_stat_annotation(ax, data=plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside',line_offset=0.1, line_offset_to_box=0.005, verbose=0)
ax.legend(title='Test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/{}_{}_model_comparison.png".format(error, Trained_cond), bbox_inches='tight')
plt.close()


'''Angular Error '''
error = 'Angular' # Angular or Euclidean
aov = pg.anova(dv='{}_error'.format(error), between=['model', 'test_cond'], data=plot_data,
             detailed=True)
print(aov)
postdoc =plot_data.pairwise_ttests(dv='{}_error'.format(error),
                                   between=['model', 'test_cond'],
                                   padjust='fdr_bh',
                                   parametric=True).round(3)
sig_results = postdoc[postdoc['p-corr']<0.05]

sns_setup_small(sns, (8,6))
ax = sns.barplot(data = plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond')
ax.set(xlabel='', ylabel='{} Error'.format(error), title='Transformer Trained: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
              (('Transformer','floating heads'),('Transformer','headless bodies'))]
ps = [0.001, 0.001,0.001, 0.001,0.029]
add_stat_annotation(ax, data=plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside',line_offset=0.1, line_offset_to_box=0.005, verbose=0)
ax.legend(title='Test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/{}_{}_model_comparison.png".format(error, Trained_cond), bbox_inches='tight')
plt.close()
