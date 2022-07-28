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

image_info = pd.read_excel('data/GroundTruth_humanest/image_info_humanmean.xlsx')
# gazed_x and gazed_y is human average annotations

'''transformer results'''
transformer = pd.DataFrame()
for epoch in [130,160,190]:  #100,120
    results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
    for f in results:
        df = pd.read_excel(f)
        if 'TEST_intact' in f: Test_cond = 'intact'
        elif 'TEST_nb' in f: Test_cond = 'floating heads'
        elif 'TEST_nh' in f: Test_cond = 'headless bodies'
        df['test_cond'] = Test_cond
        df['image'] = df.apply(lambda r: r['image'].replace('_nh', '') if '_nh' in r['image'] else  r['image'].replace('_nb', ''), axis=1)
        df = df.drop(['gazed_x','gazed_y','gazer'],axis=1)

        # calculate error based on mean
        img_info = image_info[(image_info['test_cond'] == Test_cond) & (image_info['subj'] == 'mean')]
        df = df.merge(img_info.drop(['test_cond','gaze_start_x', 'gaze_start_y',  'subj'], axis=1), on=['image'])
        df['Euclidean_error_meanest'] = np.sqrt(
            (df['gazed_x'] - df['transformer_est_x']) ** 2 + (df['gazed_y'] - df['transformer_est_y']) ** 2)
        df['Angular_error_meanest'] = df.apply(lambda r: compute_angle(r, 'transformer'), axis=1)
        df = df.drop(['gazed_x', 'gazed_y', 'gaze_start_x', 'gaze_start_y'], axis=1)

        # calculate error based on individual
        img_info = image_info[(image_info['test_cond'] == Test_cond) & (image_info['subj'] != 'mean')]
        df = df.merge(img_info[['image', 'gaze_start_x', 'gaze_start_y', 'gazed_x', 'gazed_y','subj']], on=['image'])
        df['Euclidean_error_lou'] = np.sqrt(
            (df['gazed_x'] - df['transformer_est_x']) ** 2 + (df['gazed_y'] - df['transformer_est_y']) ** 2)
        df['Angular_error_lou'] = df.apply(lambda r: compute_angle(r, 'transformer'), axis=1)
        df = df.groupby(['test_cond','image']).mean().reset_index().drop(['gazed_x', 'gazed_y', 'gaze_start_x', 'gaze_start_y'], axis=1)

        transformer = pd.concat([transformer,df])
transformer['model'] = '{} Transformer'.format(Trained_cond)
transformer = transformer.drop(['chong_est_x','chong_est_y'],axis=1)
transformer.to_excel('data/GroundTruth_humanest/{}_Transformer_summary.xlsx'.format(Trained_cond), index=None)

