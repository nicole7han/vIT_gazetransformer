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

def compute_angle2hori(row, model):
    vector_1 = [1, 0]
    vector_2 = [row['{}_est_x'.format(model)] - row['gaze_start_x'], row['{}_est_y'.format(model)] - row['gaze_start_y']]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi  # angle in degrees
    return angle


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'


Trained_cond = 'HeadBody'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)
image_info = pd.read_excel('data/GroundTruth_gazedperson/image_info.xlsx') # gazed location information (with gazer)
image_info_humanmean = pd.read_excel('data/GroundTruth_humanest/image_info_humanmean.xlsx') #gazed location information (no gazer)

'''transformer results'''
transformer = pd.DataFrame()
for epoch in [300,100,120]: #100,120
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'
        df['test_cond'] = Test_cond
        transformer = pd.concat([transformer,df])
transformer = transformer.drop(['gazed_x', 'gazed_y', 'chong_est_x','chong_est_y'],axis=1)
transformer['image'] = transformer.apply(cleanimagename, axis=1)
transformer = transformer.merge(image_info_humanmean.drop([ 'gaze_start_x','gaze_start_y'], axis=1),
                                on=['test_cond','image'])
# image_info.to_excel('data/GroundTruth_gazedperson/image_info.xlsx', index=None)
transformer['Angle2Hori'] = transformer.apply(lambda r: compute_angle2hori(r, 'transformer'), axis=1)
transformer = transformer[['test_cond', 'image', 'gazer', 'Angle2Hori', 'gaze_start_x','gaze_start_y','transformer_est_x', 'transformer_est_y', 'gazed_x','gazed_y',]]
transformer['model'] = '{} Transformer'.format(Trained_cond)
transformer.to_excel('data/GroundTruth_humanest/{}_Transformer_vectors.xlsx'.format(Trained_cond), index=None)



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
    # df = df.merge(image_info, on=['image'])
    df = df[(df['subj'] != 99401) & (df['subj'] != 99807)]
    df['test_cond'] = Test_cond
    humans = pd.concat([humans,df])

humans = humans.merge(image_info, on=['test_cond','image','gazer'])
humans['Angle2Hori'] = humans.apply(lambda r: compute_angle2hori(r, 'human'), axis=1)
humans = humans[['test_cond', 'image', 'gazer', 'subj', 'Angle2Hori','human_est_x', 'human_est_y','gazed_x','gazed_y',]]
humans['model'] = 'Humans'

humans.to_excel('data/GroundTruth_humanest/Humans_vectors.xlsx'.format(Trained_cond), index=None)




'''CNN results'''
results = glob.glob('{}/chong*.csv'.format(basepath))
cnn = pd.DataFrame()
for f in results:
    df = pd.read_csv(f)
    if 'intact' in f: Test_cond = 'intact'
    elif 'nb' in f: Test_cond = 'floating heads'
    elif 'nh' in f: Test_cond = 'headless bodies'
    df['test_cond'] = Test_cond
    cnn = pd.concat([cnn,df])
cnn = cnn.drop([ 'gaze_start_x', 'gaze_start_y'], axis=1)
cnn = cnn.merge(image_info, on=['image','gazer'])
cnn['Angle2Hori'] = cnn.apply(lambda r: compute_angle2hori(r, 'chong'), axis=1)
cnn = cnn[['test_cond', 'image', 'gazer', 'Angle2Hori', 'gaze_start_x','gaze_start_y','chong_est_x', 'chong_est_y','gazed_x','gazed_y',]]
cnn['model'] = 'CNN'
cnn.to_excel('data/GroundTruth_humanest/CNN_vectors.xlsx'.format(Trained_cond), index=None)


# plot_data = pd.concat([transformer, cnn, humans])
# plot_data['test_cond'] = plot_data['test_cond'].astype('category')
# plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
# plot_data.to_excel('data/GroundTruth_gazedperson/{}_summary.xlsx'.format(Trained_cond), index=None)
