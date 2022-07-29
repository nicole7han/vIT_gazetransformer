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
        df['test_cond'] = Test_cond
        transformer = pd.concat([transformer,df])
# COMPUTE HUMAN AND MODEL VECTOR
image_info = transformer[['image','gazer','gaze_start_x','gaze_start_y','gazed_x','gazed_y']].drop_duplicates()
# image_info.to_excel('data/GroundTruth_gazedperson/image_info.xlsx', index=None)
transformer['Angle2Hori'] = transformer.apply(lambda r: compute_angle2hori(r, 'transformer'), axis=1)
transformer = transformer[['test_cond', 'image', 'gazer', 'Angle2Hori', 'gaze_start_x','gaze_start_y','transformer_est_x', 'transformer_est_y', 'gazed_x','gazed_y',]]
transformer.to_excel('data/GroundTruth_gazedperson/{}_Transformer_vectors.xlsx'.format(Trained_cond), index=None)

