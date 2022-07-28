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

Trained_cond = 'HeadBody'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)

# human_estimations = pd.read_excel('data/Human_estimations.xlsx')
# # calculate error with respective to mean humans
# mean_gt = human_estimations.groupby(['test_cond','image']).mean().reset_index()[
#     ['test_cond','image', 'human_est_x', 'human_est_y', 'gaze_start_x', 'gaze_start_y']]
# mean_gt.columns = ['test_cond','image', 'gazed_x', 'gazed_y', 'gaze_start_x', 'gaze_start_y']
# mean_gt['subj'] = 'mean'
#
#
# # calculate error with respective to all the rest humans (leave one out)
# ind_gt = pd.DataFrame()
# for test_cond in ['intact','floating heads','headless bodies']:
#     print(test_cond)
#     for subj in np.unique(human_estimations['subj']):
#         rest = human_estimations[(human_estimations['test_cond']==test_cond) & (human_estimations['subj']!=subj)]
#         rest = rest.groupby('image').mean().reset_index()[
#         ['image', 'human_est_x', 'human_est_y', 'gaze_start_x', 'gaze_start_y']]
#         rest.columns = ['image', 'gazed_x', 'gazed_y', 'gaze_start_x', 'gaze_start_y']
#         rest['test_cond'] = test_cond
#         rest['subj'] = subj # this is the gt for this particular subject (averaged using leave one out)
#         ind_gt = ind_gt.append(rest, ignore_index=True)
#
# image_info_humanmean = pd.concat([mean_gt,ind_gt])
# image_info_humanmean.to_excel('data/GroundTruth_gazedperson/image_info_humanmean.xlsx', index=None)



image_info = pd.read_excel('data/GroundTruth_humanest/image_info_humanmean.xlsx')
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
    df = df[(df['subj'] != 99401) & (df['subj'] != 99807)]
    df['test_cond'] = Test_cond

    # calculate error based on mean
    img_info = image_info[(image_info['test_cond']==Test_cond) & (image_info['subj']=='mean')]
    df = df.merge(img_info.drop(['test_cond','subj'],axis=1), on=['image'])
    df['Euclidean_error_meanest'] = np.sqrt(
        (df['gazed_x'] - df['human_est_x']) ** 2 + (df['gazed_y'] - df['human_est_y']) ** 2)
    df['Angular_error_meanest'] = df.apply(lambda r: compute_angle(r, 'human'), axis=1)
    df = df.drop(['gazed_x','gazed_y','gaze_start_x','gaze_start_y'], axis=1)

    # calculate error based on individual
    img_info = image_info[(image_info['test_cond']==Test_cond) & (image_info['subj']!='mean')]
    df = df.merge(img_info.drop(['test_cond'],axis=1), on=['image','subj'])
    df['Euclidean_error_lou'] = np.sqrt(
        (df['gazed_x'] - df['human_est_x']) ** 2 + (df['gazed_y'] - df['human_est_y']) ** 2)
    df['Angular_error_lou'] = df.apply(lambda r: compute_angle(r, 'human'), axis=1)
    df = df.drop(['gazed_x','gazed_y','gaze_start_x','gaze_start_y'], axis=1)
    humans = pd.concat([humans,df])
humans['model'] = 'Humans'



'''transformer results'''
transformer = pd.DataFrame()
for epoch in [300,100,120]: #100,120
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'
        df = df.groupby('image').mean().reset_index()  # compute estimation for each image
        df['test_cond'] = Test_cond
        df['image'] = df.apply(lambda r: r['image'].replace('_nh', '') if '_nh' in r['image'] else  r['image'].replace('_nb', ''), axis=1)
        df = df.drop(['gazed_x','gazed_y','gazer'],axis=1)

        # calculate error based on mean
        img_info = image_info[(image_info['test_cond'] == Test_cond) & (image_info['subj'] == 'mean')]
        df = df.merge(img_info[['image', 'gazed_x', 'gazed_y']], on=['image'])
        df['Euclidean_error_meanest'] = np.sqrt(
            (df['gazed_x'] - df['transformer_est_x']) ** 2 + (df['gazed_y'] - df['transformer_est_y']) ** 2)
        df['Angular_error_meanest'] = df.apply(lambda r: compute_angle(r, 'transformer'), axis=1)
        df = df.drop(['gazed_x', 'gazed_y'], axis=1)

        # calculate error based on individual
        img_info = image_info[(image_info['test_cond'] == Test_cond) & (image_info['subj'] != 'mean')]
        df = df.merge(img_info[['image', 'gazed_x', 'gazed_y','subj']], on=['image'])
        df['Euclidean_error_lou'] = np.sqrt(
            (df['gazed_x'] - df['transformer_est_x']) ** 2 + (df['gazed_y'] - df['transformer_est_y']) ** 2)
        df['Angular_error_lou'] = df.apply(lambda r: compute_angle(r, 'transformer'), axis=1)
        df = df.groupby(['test_cond','image']).mean().reset_index().drop(['gazed_x', 'gazed_y', 'gaze_start_x', 'gaze_start_y'], axis=1)

        transformer = pd.concat([transformer,df])
transformer['model']='Transformer'
transformer = transformer.drop(['chong_est_x','chong_est_y'],axis=1)
image_info_gazer = transformer[['image','gaze_start_x','gaze_start_y']].drop_duplicates()
image_info = image_info_gazer.merge(image_info_gazer, on='image')


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
cnn = cnn[['image', 'test_cond','Euclidean_error','Angular_error']]
cnn['model'] = 'CNN'

plot_data = pd.concat([transformer, cnn, humans])
plot_data['test_cond'] = plot_data['test_cond'].astype('category')
plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
plot_data.to_excel('data/{}_summary.xlsx'.format(Trained_cond), index=None)


transformer['Euclidean_error'] = np.sqrt( (transformer['gazed_x']-transformer['transformer_est_x'])**2 + (transformer['gazed_y']-transformer['transformer_est_y'])**2 )
transformer['Angular_error'] = transformer.apply(lambda r: compute_angle(r,'transformer'),axis=1)
transformer = transformer.groupby(['image','test_cond']).mean().reset_index()
transformer = transformer[['image', 'test_cond','Euclidean_error','Angular_error']]
transformer['model'] = 'Transformer'









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
