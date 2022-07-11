import pandas as pd
import glob
from script.model import *
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
setpallet = sns.color_palette("Set2")

basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
epoch=20
checkpoint = torch.load('trainedmodels/model_body_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
# plt.plot(checkpoint['train_loss'])
# plt.plot(checkpoint['test_loss'])
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)
model.to(device)


Trained_cond = 'Body'
outpath = '{}/model_eval_viu_outputs/Trained_{}'.format(basepath,Trained_cond)

'''transformer results'''
results = glob.glob('{}/*.xlsx'.format(outpath))
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
results = glob.glob('{}/*.csv'.format(basepath))
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




plot_data = pd.concat([transformer, cnn])
plot_data['test_cond'] = plot_data['test_cond'].astype('category')
plot_data['test_cond'].cat.reorder_categories(['intact', 'floating heads', 'headless bodies'], inplace=True)
sns_setup_small(sns, (9,7))
ax = sns.barplot(data = plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond')
ax.set(xlabel='', ylabel='Euclidean Error', title='Trained condition: {}'.format(Trained_cond))
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(title='Test condition', frameon=False,loc='upper right', bbox_to_anchor=(1.35, 1.05))
ax.figure.savefig("figures/{}_{}_model_comparison.jpg".format(Trained_cond, epoch), bbox_inches='tight')
plt.close()