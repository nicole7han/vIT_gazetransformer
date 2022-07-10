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
epoch=182
checkpoint = torch.load('trainedmodels/model_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
# plt.plot(checkpoint['train_loss'])
# plt.plot(checkpoint['test_loss'])
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)
model.to(device)


Trained_cond = 'Intact'
outpath = '{}/model_eval_viu_outputs/Trained{}'.format(basepath,Trained_cond)
results = glob.glob('{}/*.xlsx'.format(outpath))
allresult = pd.DataFrame()
for f in results:
    df = pd.read_excel(f)
    if 'TEST_intact' in f: Test_cond = 'intact'
    elif 'TEST_nb' in f: Test_cond = 'floating heads'
    elif 'TEST_nh' in f: Test_cond = 'headless bodies'

    df['test_cond'] = Test_cond
    allresult = pd.concat([allresult,df])

allresult['transformer_Euclidean_error'] = np.sqrt( (allresult['gazed_x']-allresult['transformer_est_x'])**2 + (allresult['gazed_y']-allresult['transformer_est_y'])**2 )
allresult['CNN_Euclidean_error'] = np.sqrt( (allresult['gazed_x']-allresult['chong_est_x'])**2 + (allresult['gazed_y']-allresult['chong_est_y'])**2 )
transformer = allresult[['test_cond','transformer_Euclidean_error']]
transformer.columns = ['test_cond','Euclidean_error']
transformer['model'] = 'transformer'
cnn = allresult[['test_cond','CNN_Euclidean_error']]
cnn = cnn.dropna()
cnn.columns = ['test_cond','Euclidean_error']
cnn['model'] = 'CNN'
plot_data = pd.concat([transformer, cnn])



sns_setup_small(sns, (9,7))
ax = sns.barplot(data = plot_data, x = 'model', y = 'Euclidean_error', hue='test_cond')
ax.set(xlabel='', ylabel='Euclidean Error', title='Trained condition: Intact')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend(title='Test condition', frameon=False,loc='upper right', bbox_to_anchor=(1.05, 1.05))
