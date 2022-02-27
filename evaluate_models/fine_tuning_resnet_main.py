import pandas as pd
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *

basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
epoch=114
checkpoint = torch.load('trainedmodels/resviTmodel_epoch{}.pt'.format(epoch), map_location='cpu')
plt.plot(checkpoint['train_loss'])
# checkpoint['test_loss']
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)
model.to(device)

# #fine tuning model on viu dataset
# train(e_start, num_e, anno_path, train_img_path, train_bbx_path, test_img_path, test_bbx_path)

# # evaluate model on viu dataset
# model = Gaze_Transformer()
# checkpoint = torch.load('models/model_epoch{}.pt'.format(epoch), map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# output = evaluate_model_gaze(anno_path, test_img_path, test_bbx_path, model, fig_path)
# output.to_excel('gaze_video_data/model_epoch{}_result.xlsx'.format(epoch))

# evaluate both models' estimation on viu dataset
datapath = "/Users/nicolehan/Documents/Research/gazetransformer/gaze_video_data"
anno_path = '{}/Video_Info.xlsx'.format(datapath)
# test_img_path = "{}/transformer_all_img_intact".format(datapath)
# test_bbx_path = "{}/transformer_all_bbx".format(datapath)
# train_img_path='gaze_video_data/transformer_train_img'
# train_bbx_path='gaze_video_data/transformer_train_bbx'
test_img_path='{}/transformer_all_img_intact'.format(datapath)
test_bbx_path='{}/transformer_all_bbx'.format(datapath)
if 'nb' in test_img_path: cond='nb'
elif 'nh' in test_img_path : cond='nh'
else: cond='intact'
head_bbx_path='{}/boundingbox_head (gaze-orienting people)'.format(datapath)
fig_path='{}/viT_epoch{}_{}_outputs'.format(datapath,epoch,cond)

try:
 chong_est = pd.read_excel('{}/Chong_model_estimation.xlsx'.format(datapath))
 output = evaluate_2model(anno_path, test_img_path, test_bbx_path, head_bbx_path, chong_est, model, fig_path,
                          bbx_noise=False)
except:
 output = evaluate_2model(anno_path, test_img_path, test_bbx_path, head_bbx_path, None, model, fig_path,
                          bbx_noise=False)
output.to_excel('{}/chong&transformer_epoch{}_{}_result.xlsx'.format(datapath,epoch,cond))
analyze_error(output, epoch, filename=datapath)


## compare intact, floating heads and headless bodies
intact = pd.read_excel('{}/chong&transformer_epoch{}_intact_result.xlsx'.format(datapath,epoch))
fh = pd.read_excel('{}/chong&transformer_epoch{}_nb_result.xlsx'.format(datapath,epoch))
hb = pd.read_excel('{}/chong&transformer_epoch{}_nh_result.xlsx'.format(datapath,epoch))
intact['condition'] = 'intact'
fh['condition'] = 'floating heads'
hb['condition'] = 'headless bodies'
alldata = pd.concat([intact, fh, hb])

alldata['transformer_eucli_error'] = np.sqrt((alldata['gazed_x'] - alldata['transformer_est_x']) ** 2 + (
         alldata['gazed_y'] - alldata['transformer_est_y']) ** 2)
alldata['transformer_ang_error'] = alldata.apply(lambda row: compute_angle(row, 'transformer'), axis=1)

sns_setup(sns)
model = ols('transformer_eucli_error ~ C(condition)', data=alldata).fit()
ax = sns.barplot(data=alldata, x='condition', y='transformer_eucli_error')
ax.set(xlabel='', ylabel='Euclidean Error')
change_width(ax,.4)
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig('{}/Figures/error_epoch{}.jpg'.format(datapath,epoch))

sns_setup(sns)
ax = sns.barplot(data=alldata, x='condition', y='transformer_ang_error')
ax.set(xlabel='', ylabel='Angular Error')
change_width(ax,.4)
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.figure.savefig('{}/Figures/angerror_epoch{}.jpg'.format(datapath,epoch))


# # test on the test images from large dataset
 datapath = os.path.abspath(os.curdir)
 anno_path = "{}/data/annotations".format(datapath)
 test_img_path = "{}/data/test".format(datapath)
 test_bbx_path = "{}/data/test_bbox".format(datapath)
 fig_path = '{}/model_eval_outputs_small/resnet_epoch{}_eval_outputs'.format(datapath, epoch)
 criterion = nn.MSELoss(reduction='mean')
 chong_est = pd.read_excel('{}/data/Chong_estimation_test.xlsx'.format(datapath))
 output = evaluate_test(anno_path, test_img_path, test_bbx_path, chong_est, criterion,  model, fig_path)
 output.to_excel('{}/model_eval_outputs_small/transformer_epoch{}_result.xlsx'.format(datapath, epoch))
 analyze_error(output, epoch, filename='figures/model_largedata')