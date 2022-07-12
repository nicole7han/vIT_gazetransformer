import pandas as pd
from script.model import *
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *

basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
epoch=178
checkpoint = torch.load('trainedmodels/model_body_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
plt.plot(checkpoint['train_loss'])
plt.plot(checkpoint['test_loss'])
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)
model.to(device)


# output = evaluate_model_gaze(anno_path, test_img_path, test_bbx_path, model, fig_path)
# output.to_excel('gaze_video_data/model_epoch{}_result.xlsx'.format(epoch))

# evaluate both models' estimation on viu dataset
datapath = "{}/gaze_video_data".format(basepath)
outpath = '{}/model_eval_viu_outputs/Trained_Body'.format(basepath)
os.makedirs(outpath, exist_ok=True)
anno_path = '{}/Video_Info.xlsx'.format(datapath)
for cond in ['intact','nb','nh']:
    test_img_path = "{}/transformer_all_img_{}".format(datapath,cond)
    test_bbx_path = "{}/transformer_all_bbx".format(datapath)
    if cond == 'intact':
        gazer_bbox = 'hb' # indicate what bounding box needs to feed into the model
    elif cond == 'nb':
        gazer_bbox = 'h'
    elif cond == 'nh':
        gazer_bbox = 'b'
    fig_path='{}/transformer_TEST_{}_epoch{}'.format(outpath,cond,epoch)

    matcher = build_matcher(set_cost_class=1, set_cost_bbox=5, set_cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 20, 'loss_giou': 2}
    losses = ['labels', 'boxes']
    num_classes = 1
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.01, losses=losses)
    # chong model test and trained on just head
    if cond == 'intact':
        chong_est = pd.read_csv('{}/chong_estimation_intact.csv'.format(basepath))
    else: 
        chong_est = None

    output = evaluate_2model(anno_path, test_img_path, test_bbx_path, chong_est, model, fig_path, criterion,
                        bbx_noise=False, gazer_bbox=gazer_bbox, cond=cond)

    output.to_excel('{}/transformer_TEST_{}_epoch{}_result.xlsx'.format(outpath,cond,epoch), index=None)
    analyze_error(output, epoch, path=outpath, cond=cond)
