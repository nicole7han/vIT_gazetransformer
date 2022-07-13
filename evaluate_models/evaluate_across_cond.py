import pandas as pd
import glob
from script.model import *
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
# model = Gaze_Transformer()
epoch=102
# checkpoint = torch.load('trainedmodels/model_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
# # plt.plot(checkpoint['train_loss'])
# # plt.plot(checkpoint['test_loss'])
# loaded_dict = checkpoint['model_state_dict']
# prefix = 'module.'
# n_clip = len(prefix)
# adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
#                 if k.startswith(prefix)}
# model.load_state_dict(adapted_dict)
# model.to(device)


Trained_cond = 'HeadBody'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)

'''transformer results'''
results = glob.glob('{}/*{}_result.xlsx'.format(outpath,epoch))
transformer = pd.DataFrame()
for f in results:
    df = pd.read_excel(f)
    if 'TEST_intact' in f: Test_cond = 'intact'
    elif 'TEST_nb' in f: Test_cond = 'floating heads'
    elif 'TEST_nh' in f: Test_cond = 'headless bodies'

    df['test_cond'] = Test_cond
    transformer = pd.concat([transformer,df])
image_info = transformer[['image','gazed_x','gazed_y']].drop_duplicates()
transformer['Euclidean_error'] = np.sqrt( (transformer['gazed_x']-transformer['transformer_est_x'])**2 + (transformer['gazed_y']-transformer['transformer_est_y'])**2 )
transformer = transformer[['test_cond','Euclidean_error']]
transformer['model'] = 'transformer'

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

cnn = cnn.merge(image_info, on=['image'])
cnn['Euclidean_error'] = np.sqrt( (cnn['gazed_x']-cnn['chong_est_x'])**2 + (cnn['gazed_y']-cnn['chong_est_y'])**2 )
cnn = cnn[['test_cond','Euclidean_error']]
cnn['model'] = 'cnn'


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
humans['model'] = 'humans'


plot_data = pd.concat([transformer, cnn, humans])
plot_data['test_cond'] = plot_data['test_cond'].astype('category')
plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
sns_setup_small(sns, (9,7))
ax = sns.barplot(data = plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond')
ax.set(xlabel='', ylabel='Euclidean Error', title='Transformer Trained: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(title='Test condition', frameon=False,loc='upper left')
ax.figure.savefig("figures/{}_{}_model_comparison.jpg".format(Trained_cond, epoch), bbox_inches='tight')
plt.close()