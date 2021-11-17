import pandas as pd

from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *

basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
epoch=8
checkpoint = torch.load('trainedmodels/resviTmodel_epoch{}.pt'.format(epoch), map_location='cpu')
checkpoint['train_loss']
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

# # evaluate both models' estimation on viu dataset
# anno_path = 'gaze_video_data/Video_Info.xlsx'
# train_img_path='gaze_video_data/transformer_train_img'
# train_bbx_path='gaze_video_data/transformer_train_bbx'
# test_img_path='gaze_video_data/transformer_all_img'
# test_bbx_path='gaze_video_data/transformer_all_bbx'
# head_bbx_path='gaze_video_data/boundingbox_head (gaze-orienting people)'
# chong_est = pd.read_excel('gaze_video_data/Chong_model_estimation.xlsx')
# fig_path='gaze_video_data/viT_epoch{}_eval_outputs'.format(epoch)

# output = evaluate_2model(anno_path, test_img_path, test_bbx_path, head_bbx_path, chong_est, model, fig_path, bbx_noise=False)
# output.to_excel('gaze_video_data/chong&transformer_epoch{}_result.xlsx'.format(epoch))
# analyze_error(output, epoch)


# test on the test images from large dataset
datapath = os.path.abspath(os.curdir)
anno_path = "{}/data/annotations".format(datapath)
test_img_path = "{}/data/test".format(datapath)
test_bbx_path = "{}/data/test_bbox".format(datapath)
fig_path = '{}/model_eval_outputs_small/resnet_epoch{}_eval_outputs'.format(datapath, epoch)
criterion = nn.MSELoss(reduction='mean')
chong_est = pd.read_excel('{}/data/Chong_estimation_test.xlsx'.format(datapath))
output = evaluate_test(anno_path, test_img_path, test_bbx_path, chong_est, criterion,  model, fig_path)
output.to_excel('{}/model_eval_outputs_small/transformer_epoch{}_result.xlsx'.format(datapath, epoch))
analyze_error(output, epoch, filename='model_largedata')