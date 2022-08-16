import pandas as pd
import glob, random
from script.model import *
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statannot import add_stat_annotation
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
from evaluate_models.utils import *

setpallet = sns.color_palette("Set2")

def compute_angle(row, model):
    vector_1 = [row['gazed_x'] - row['gaze_start_x'], row['gazed_y'] - row['gaze_start_y']]
    vector_2 = [row['{}_est_x'.format(model)] - row['gaze_start_x'], row['{}_est_y'.format(model)] - row['gaze_start_y']]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi  # angle in degrees
    return angle

def compute_angle2hori(row, model):
    vector_1 = [1, 0]
    vector_2 = [row['{}_est_x'.format(model)] - row['gaze_start_x'], row['{}_est_y'.format(model)] - row['gaze_start_y']]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi  # angle in degrees
    return angle


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
Trained_cond = 'HeadBody_NegSamples'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)
N_perm = 10000 # number of permutations

'''transformer results'''
transformer = pd.DataFrame()
for epoch in [20]: #300,100,340
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        df['image'] = df.apply(cleanimagename, axis=1)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'

        df_estmean = df.groupby('image').mean().reset_index()  # average across gazers
        # important to keep individual gazer for calculating angular error
        df_estmean = df_estmean[['image','transformer_est_x','transformer_est_y']]
        df_estmean.columns = ['image','transformermean_est_x','transformermean_est_y']
        df = df.merge(df_estmean,on='image')
        df['test_cond'] = Test_cond
        transformer = pd.concat([transformer,df])


image_info = transformer[['image','gazer','gaze_start_x','gaze_start_y','gazed_x','gazed_y']].drop_duplicates()
image_info = image_info.drop_duplicates()
# image_info.to_excel('data/GroundTruth_gazedperson/image_info.xlsx', index=None)
transformer['Euclidean_error'] = np.sqrt( (transformer['gazed_x']-transformer['transformermean_est_x'])**2 + (transformer['gazed_y']-transformer['transformermean_est_y'])**2 )
transformer['Angular_error'] = transformer.apply(lambda r: compute_angle(r,'transformermean'),axis=1)
transformer = transformer.groupby(['image','test_cond']).mean().reset_index()
transformer = transformer[['image', 'test_cond','Euclidean_error','Angular_error']]
transformer['model'] = 'HeadBody Transformer'
transformer.to_excel('data/GroundTruth_gazedperson/{}_Transformer_summary.xlsx'.format(Trained_cond), index=None)

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
perm['model'] = '{} Transformer'.format()
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





'''CNN results'''
results = glob.glob('{}/chong*.csv'.format(basepath))
cnn = pd.DataFrame()
for f in results:
    df = pd.read_csv(f)
    if 'intact' in f: Test_cond = 'intact'
    elif 'nb' in f: Test_cond = 'floating heads'
    elif 'nh' in f: Test_cond = 'headless bodies'
    df_estmean = df.groupby('image').mean().reset_index() # average across gazers
    # important to keep individual gazer for calculating angular error
    df_estmean = df_estmean[['image','chong_est_x','chong_est_y']]
    df_estmean.columns = ['image','chongmean_est_x','chongmean_est_y']
    df = df.merge(df_estmean,on='image')

    df['test_cond'] = Test_cond
    cnn = pd.concat([cnn,df])

cnn = cnn.merge(image_info[['image', 'gazer','gazed_x', 'gazed_y']], on=['image','gazer'])
cnn['Euclidean_error'] = np.sqrt( (cnn['gazed_x']-cnn['chongmean_est_x'])**2 + (cnn['gazed_y']-cnn['chongmean_est_y'])**2 )
cnn['Angular_error'] = cnn.apply(lambda r: compute_angle(r,'chongmean'),axis=1)
cnn = cnn.groupby(['image','test_cond']).mean().reset_index()
cnn = cnn[['image', 'test_cond','Euclidean_error','Angular_error']]
cnn['model'] = 'Head CNN'
cnn.to_excel('data/GroundTruth_gazedperson/CNN_summary.xlsx'.format(Trained_cond), index=None)


# permutation error
perm = pd.DataFrame()
for _ in range(N_perm):
    cnn_perm = cnn.copy()
    cnn_perm['estxy'] = cnn_perm['chongmean_est_x'].astype(str) + ',' +\
                                 cnn_perm['chongmean_est_y'].astype(str)
    cnn_perm['estxy_perm'] = random.sample(list(cnn_perm['estxy']), len(cnn_perm))
    cnn_perm[['chongmean_est_x','chongmean_est_y']] = \
        cnn_perm['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    cnn_perm['Euclidean_error'] = np.sqrt((cnn_perm['gazed_x'] - cnn_perm['chongmean_est_x']) ** 2 + (
                cnn_perm['gazed_y'] - cnn_perm['chongmean_est_y']) ** 2)
    cnn_perm = cnn_perm.groupby(['image', 'test_cond']).mean().reset_index()
    cnn_perm['Angular_error'] = cnn_perm.apply(lambda r: compute_angle(r, 'chongmean'), axis=1)

    temp = cnn_perm.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]
    perm = perm.append(temp, ignore_index=True)
perm['model'] = 'Head CNN'
perm.to_excel('data/GroundTruth_gazedperson/CNN_Perm.xlsx', index=None)


# permutation error left and right
perm = pd.DataFrame()
for _ in range(N_perm):
    cnn_perm = cnn.copy()
    cnn_perm['gazed_wrt_gazer'] = cnn_perm.apply(lambda r: 1 if r['gazed_x'] > r['gaze_start_x'] else -1, axis=1)
    # permutate within right and left subgroups
    left = cnn_perm[cnn_perm['gazed_wrt_gazer']==-1]
    right = cnn_perm[cnn_perm['gazed_wrt_gazer']==1]

    left['estxy'] = left['chongmean_est_x'].astype(str) + ',' +\
                                 left['chongmean_est_y'].astype(str)
    left['estxy_perm'] = random.sample(list(left['estxy']), len(left))
    left[['chongmean_est_x','chongmean_est_y']] = \
        left['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    left['Euclidean_error'] = np.sqrt((left['gazed_x'] - left['chongmean_est_x']) ** 2 + (
                left['gazed_y'] - left['chongmean_est_y']) ** 2)
    left = left.groupby(['image', 'test_cond']).mean().reset_index()
    left['Angular_error'] = left.apply(lambda r: compute_angle(r, 'chongmean'), axis=1)
    left = left.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]

    right['estxy'] = right['chongmean_est_x'].astype(str) + ',' +\
                                 right['chongmean_est_y'].astype(str)
    right['estxy_perm'] = random.sample(list(right['estxy']), len(right))
    right[['chongmean_est_x','chongmean_est_y']] = \
        right['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    right['Euclidean_error'] = np.sqrt((right['gazed_x'] - right['chongmean_est_x']) ** 2 + (
                right['gazed_y'] - right['chongmean_est_y']) ** 2)
    right = right.groupby(['image', 'test_cond']).mean().reset_index()
    right['Angular_error'] = right.apply(lambda r: compute_angle(r, 'chongmean'), axis=1)
    right = right.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]

    perm = perm.append(pd.concat([left, right]), ignore_index=True)
perm['model'] = 'Head CNN'
perm.to_excel('data/GroundTruth_gazedperson/Head_CNN_leftright_Perm.xlsx', index=None)



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
    df = df[(df['subj'] != 99401) & (df['subj'] != 99807)]
    df['test_cond'] = Test_cond
    humans = pd.concat([humans,df])

humans['Euclidean_error'] = np.sqrt(
    (humans['gazed_x'] - humans['human_est_x']) ** 2 + (humans['gazed_y'] - humans['human_est_y']) ** 2)
humans['Angular_error'] = humans.apply(lambda r: compute_angle(r,'human'),axis=1)
humans = humans.groupby(['image','test_cond']).mean().reset_index()
humans = humans[['image', 'test_cond','Euclidean_error','Angular_error']]
humans['model'] = 'Humans'

humans.to_excel('data/GroundTruth_gazedperson/Humans_summary.xlsx'.format(Trained_cond), index=None)
#
# plot_data = pd.concat([transformer, cnn, humans])
# plot_data['test_cond'] = plot_data['test_cond'].astype('category')
# plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
# plot_data.to_excel('data/GroundTruth_gazedperson/{}_summary.xlsx'.format(Trained_cond), index=None)

# permutation error
perm = pd.DataFrame()
for _ in range(N_perm):
    human_perm = humans.copy()
    human_perm['estxy'] = human_perm['human_est_x'].astype(str) + ',' +\
                                 human_perm['human_est_y'].astype(str)
    human_perm['estxy_perm'] = random.sample(list(human_perm['estxy']), len(human_perm))
    human_perm[['human_est_x','human_est_y']] = \
        human_perm['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    human_perm['Euclidean_error'] = np.sqrt((human_perm['gazed_x'] - human_perm['human_est_x']) ** 2 + (
                human_perm['gazed_y'] - human_perm['human_est_y']) ** 2)
    human_perm = human_perm.groupby(['image', 'test_cond']).mean().reset_index()
    human_perm['Angular_error'] = human_perm.apply(lambda r: compute_angle(r, 'human'), axis=1)

    temp = human_perm.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]
    perm = perm.append(temp, ignore_index=True)
perm['model'] = 'Humans'
perm.to_excel('data/GroundTruth_gazedperson/Humans_Perm.xlsx'.format(Trained_cond), index=None)

# permutation error left and right
perm = pd.DataFrame()
for _ in range(N_perm):
    human_perm = humans.copy()
    human_perm['gazed_wrt_gazer'] = human_perm.apply(lambda r: 1 if r['gazed_x'] > r['gaze_start_x'] else -1, axis=1)
    # permutate within right and left subgroups
    left = human_perm[human_perm['gazed_wrt_gazer']==-1]
    right = human_perm[human_perm['gazed_wrt_gazer']==1]

    left['estxy'] = left['human_est_x'].astype(str) + ',' +\
                                 left['human_est_y'].astype(str)
    left['estxy_perm'] = random.sample(list(left['estxy']), len(left))
    left[['human_est_x','human_est_y']] = \
        left['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    left['Euclidean_error'] = np.sqrt((left['gazed_x'] - left['human_est_x']) ** 2 + (
                left['gazed_y'] - left['human_est_y']) ** 2)
    left = left.groupby(['image', 'test_cond']).mean().reset_index()
    left['Angular_error'] = left.apply(lambda r: compute_angle(r, 'human'), axis=1)
    left = left.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]


    right['estxy'] = right['human_est_x'].astype(str) + ',' +\
                                 right['human_est_y'].astype(str)
    right['estxy_perm'] = random.sample(list(right['estxy']), len(right))
    right[['human_est_x','human_est_y']] = \
        right['estxy_perm'].str.split(',',expand=True).astype('float') # update estimation with permiutations
    right['Euclidean_error'] = np.sqrt((right['gazed_x'] - right['human_est_x']) ** 2 + (
                right['gazed_y'] - right['human_est_y']) ** 2)
    right = right.groupby(['image', 'test_cond']).mean().reset_index()
    right['Angular_error'] = right.apply(lambda r: compute_angle(r, 'human'), axis=1)
    right = right.groupby('test_cond').mean().reset_index()[['test_cond', 'Euclidean_error', 'Angular_error']]

    perm = perm.append(pd.concat([left, right]), ignore_index=True)
perm['model'] = 'Humans'
perm.to_excel('data/GroundTruth_gazedperson/Humans_leftright_Perm.xlsx', index=None)




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
ax.set(xlabel='', ylabel='{} Error (normalized)'.format(error), title='Transformer Trained: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
             (('Transformer','intact'),('Transformer','headless bodies'))]
ps = [0.001, 0.001,0.001, 0.001,0.045]
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
ax.set(xlabel='', ylabel='{} Error (˚)'.format(error), title='Transformer Trained: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
box_pairs = [(('CNN','headless bodies'),('CNN','floating heads')),(('CNN','headless bodies'),('CNN','intact')),
             (('Humans','headless bodies'),('Humans','floating heads')),(('Humans','headless bodies'),('Humans','intact')),
              ]
ps = [0.001, 0.001,0.001, 0.001]
add_stat_annotation(ax, data=plot_data, x = 'model', y = '{}_error'.format(error), hue='test_cond',
                    box_pairs= box_pairs, perform_stat_test=False, pvalues=ps,
                    loc='outside',line_offset=0.1, line_offset_to_box=0.005, verbose=0)
ax.legend(title='Test condition', loc='upper right', frameon=False, bbox_to_anchor=[1.4, 0.9])
ax.figure.savefig("figures/{}_{}_model_comparison.png".format(error, Trained_cond), bbox_inches='tight')
plt.close()
