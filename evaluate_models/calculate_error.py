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


Trained_cond = 'Body'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)

'''transformer results'''
transformer = pd.DataFrame()
for epoch in [130,160,190]: #100,120
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'

        df_estmean = df.groupby('image').mean().reset_index()  # compute estimation for each image
        # important to keep individual gazer for calculating angular error
        df_estmean = df_estmean[['image','transformer_est_x','transformer_est_y']]
        df_estmean.columns = ['image','transformermean_est_x','transformermean_est_y']
        df = df.merge(df_estmean,on='image')
        df['test_cond'] = Test_cond
        transformer = pd.concat([transformer,df])

image_info = transformer[['image','gazer','gaze_start_x','gaze_start_y','gazed_x','gazed_y']].drop_duplicates()
transformer['Euclidean_error'] = np.sqrt( (transformer['gazed_x']-transformer['transformermean_est_x'])**2 + (transformer['gazed_y']-transformer['transformermean_est_y'])**2 )
transformer['Angular_error'] = transformer.apply(lambda r: compute_angle(r,'transformermean'),axis=1)
transformer = transformer.groupby(['image','test_cond']).mean().reset_index()
transformer = transformer[['image', 'test_cond','Euclidean_error','Angular_error']]
transformer['model'] = '{} Transformer'.format(Trained_cond)

transformer.to_excel('data/GroundTruth_gazedperson/{}_Transformer_summary.xlsx'.format(Trained_cond), index=None)


N_perm=10000
# permutation error
perm = pd.DataFrame()
for _ in range(N_perm):
    transformer_perm = transformer.copy()
    transformer_perm['estxy'] = transformer_perm['transformermean_est_x'].astype(str) + ',' +\
                                 transformer_perm['transformermean_est_y'].astype(str)
    transformer_perm['estxy_perm'] = random.sample(list(transformer_perm['estxy']), len(transformer_perm))
    transformer_perm[['transformermean_est_x','transformermean_est_y']] = \
        transformer_perm['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    transformer_perm['Euclidean_error'] = np.sqrt((transformer_perm['gazed_x'] - transformer_perm['transformermean_est_x']) ** 2 + (
                transformer_perm['gazed_y'] - transformer_perm['transformermean_est_y']) ** 2)
    transformer_perm = transformer_perm.groupby(['image', 'test_cond']).mean().reset_index()
    transformer_perm['Angular_error'] = transformer_perm.apply(lambda r: compute_angle(r, 'transformermean'), axis=1)

    temp = transformer_perm.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]
    perm = perm.append(temp, ignore_index=True)
perm['model'] = '{}_Transformer'.format(Trained_cond)
perm.to_excel('data/GroundTruth_gazedperson/{}_Transformer_Perm.xlsx'.format(Trained_cond), index=None)


# permutation error left and right
perm = pd.DataFrame()
for _ in range(N_perm):
    transformer_perm = transformer.copy()
    transformer_perm['gazed_wrt_gazer'] = transformer_perm.apply(lambda r: 1 if r['gazed_x']>r['gaze_start_x'] else -1, axis=1)
    # permutate within right and left subgroups
    left = transformer_perm[transformer_perm['gazed_wrt_gazer']==-1]
    right = transformer_perm[transformer_perm['gazed_wrt_gazer']==1]

    left['estxy'] = left['transformermean_est_x'].astype(str) + ',' +\
                                 left['transformermean_est_y'].astype(str)
    left['estxy_perm'] = random.sample(list(left['estxy']), len(left))
    left[['transformermean_est_x','transformermean_est_y']] = \
        left['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    left['Euclidean_error'] = np.sqrt((left['gazed_x'] - left['transformermean_est_x']) ** 2 + (
                left['gazed_y'] - left['transformermean_est_y']) ** 2)
    left = left.groupby(['image', 'test_cond']).mean().reset_index()
    left['Angular_error'] = left.apply(lambda r: compute_angle(r, 'transformermean'), axis=1)
    left = left.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]

    right['estxy'] = right['transformermean_est_x'].astype(str) + ',' +\
                                 right['transformermean_est_y'].astype(str)
    right['estxy_perm'] = random.sample(list(right['estxy']), len(right))
    right[['transformermean_est_x','transformermean_est_y']] = \
        right['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    right['Euclidean_error'] = np.sqrt((right['gazed_x'] - right['transformermean_est_x']) ** 2 + (
                right['gazed_y'] - right['transformermean_est_y']) ** 2)
    right = right.groupby(['image', 'test_cond']).mean().reset_index()
    right['Angular_error'] = right.apply(lambda r: compute_angle(r, 'transformermean'), axis=1)
    right = right.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]

    perm = perm.append(pd.concat([left, right]), ignore_index=True)
perm['model'] = '{} Transformer'.format(Trained_cond)
perm.to_excel('data/GroundTruth_gazedperson/{}_Transformer_leftright_Perm.xlsx'.format(Trained_cond), index=None)

#
#
#
#
# plot_data = pd.read_excel('data/{}_summary.xlsx'.format(Trained_cond))
# plot_data['test_cond'] = plot_data['test_cond'].astype('category')
# plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
# plot_data['model'] = plot_data['model'].astype('category')
# plot_data['model'].cat.reorder_categories(['Humans', 'CNN', 'Transformer'], inplace=True)
#
#
# '''Euclidean Error '''
# error = 'Euclidean' # Angular or Euclidean
# aov = pg.anova(dv='{}_error'.format(error), between=['model', 'test_cond'], data=plot_data,
#              detailed=True)
# print(aov)
# postdoc =plot_data.pairwise_ttests(dv='{}_error'.format(error),
#                                    between=['model', 'test_cond'],
#                                    padjust='fdr_bh',
#                                    parametric=True).round(3)
# sig_results = postdoc[postdoc['p-corr']<0.05]
#
#
# sns_setup_small(sns, (8,6))
# ax = sns.barplot(data = plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond')
# ax.set(xlabel='', ylabel='{} Error (normalized)'.format(error), title='Transformer Trained: {}'.format(Trained_cond))
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
#              (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
#              (('Transformer','intact'),('Transformer','headless bodies'))]
# ps = [0.001, 0.001,0.001, 0.001,0.045]
# add_stat_annotation(ax, data=plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond',
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside',line_offset=0.1, line_offset_to_box=0.005, verbose=0)
# ax.legend(title='Test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
# ax.figure.savefig("figures/{}_{}_model_comparison.png".format(error, Trained_cond), bbox_inches='tight')
# plt.close()
#
#
# '''Angular Error '''
# error = 'Angular' # Angular or Euclidean
# aov = pg.anova(dv='{}_error'.format(error), between=['model', 'test_cond'], data=plot_data,
#              detailed=True)
# print(aov)
# postdoc =plot_data.pairwise_ttests(dv='{}_error'.format(error),
#                                    between=['model', 'test_cond'],
#                                    padjust='fdr_bh',
#                                    parametric=True).round(3)
# sig_results = postdoc[postdoc['p-corr']<0.05]
#
# sns_setup_small(sns, (8,6))
# ax = sns.barplot(data = plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond')
# ax.set(xlabel='', ylabel='{} Error (˚)'.format(error), title='Transformer Trained: {}'.format(Trained_cond))
# ax.spines['top'].set_color('white')
# ax.spines['right'].set_color('white')
# box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
#              (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
#               ]
# ps = [0.001, 0.001,0.001, 0.001]
# add_stat_annotation(ax, data=plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond',
#                     box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
#                     loc='outside',line_offset=0.1, line_offset_to_box=0.005, verbose=0)
# ax.legend(title='Test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
# ax.figure.savefig("figures/{}_{}_model_comparison.png".format(error, Trained_cond), bbox_inches='tight')
# plt.close()