import pandas as pd
from script.model import *
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *

def evaluate_train(anno_path, test_img_path, test_bbx_path, chong_est, criterion, model, fig_path, savefigure=True):
    '''

    @param anno_path:gazed location
    @type anno_path: str
    @param test_img_path:test image path
    @type test_img_path: str
    @param test_bbx_path:test bounding box path
    @type test_bbx_path: str
    @param chong_est: chong estimation on test images
    @param criterion: loss criterion
    @param model: the model
    @return:output
    @rtype:dataframe
    '''

    IMAGES = []
    GAZE_START = []
    GT_GAZE = []

    TRANS_PRED = []
    TRANS_ANG_LOSS = []
    TRANS_DIS_LOSS = []
    CHONG_PRED = []
    CHONG_ANG_LOSS = []
    CHONG_DIS_LOSS = []

    test_data = GazeDataloader(anno_path, test_img_path, test_bbx_path)
    test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True)
    test_dataiter = iter(test_dataloader)
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for images_name, images, flips, h_crops, masks, eye, targetgaze, displacexy in test_dataiter:
            eye, displacexy = np.array(eye), np.array(displacexy)
            targetgaze_label = np.array(targetgaze['labels'].tolist())
            targetgaze_bbx = np.array(targetgaze['boxes']) #x_cen, y_cen, width, height

            test_b_size = images.shape[0]
            gaze_pred = model(images, h_crops, masks)
            gaze_pred_logits = np.array(gaze_pred['pred_logits'].detach()) # bs x 100 x 2
            gaze_pred_bbx = np.array(gaze_pred['pred_boxes'].detach())  # bs x 100 x 4

            # loss
            targets = [{'labels': targetgaze['labels'][i][0].unsqueeze(0).to(device),
                        'boxes': targetgaze['boxes'][i].unsqueeze(0).to(device)} \
                       for i in range(test_b_size)]
            indices = np.array(criterion.matcher(gaze_pred, targets))
            for i in range(test_b_size):
                img = plt.imread(images_name[i])
                try: h, w, _ = img.shape
                except:h, w = img.shape

                # find the bbox idx out of 100 queries
                idx = indices[i][0]
                gaze_bbx = gaze_pred_bbx[i][idx]

                # transformer prediction (relative with background)
                disx, disy = displacexy[i]
                # # transform gaze_pred, targetgaze, eye position from background to image
                # trans_pred_x, trans_pred_y = coord_bg2img(gaze_bbx[0], gaze_bbx[1], disx, disy)
                # eye_x, eye_y = coord_bg2img(eye[i][0], eye[i][1], disx, disy)
                # target_x, target_y = coord_bg2img(targetgaze_bbx[i][0], targetgaze_bbx[i][1], disx, disy)
                trans_pred_x, trans_pred_y = gaze_bbx[0], gaze_bbx[1]
                eye_x, eye_y = eye[i][0], eye[i][1]
                target_x, target_y = targetgaze_bbx[i][0], targetgaze_bbx[i][1]

                # flip transformer x if the image is flipped horizontally, keep all xy in original image coordination
                if flips[i]:
                    trans_pred_x = 1- trans_pred_x
                    eye_x = 1 - eye_x
                    target_x = 1-target_x

                vec1 = (target_x - eye_x, target_y - eye_y) #groundtruth vector
                vec2 = (trans_pred_x - eye_x, trans_pred_y - eye_y) # transformer estimation

                v1, v2, = [vec1[0] * w, vec1[1] * h], \
                         [vec2[0] * w, vec2[1] * h]

                unit_vector_1 = v1 / np.linalg.norm(v1)
                unit_vector_2 = v2 / np.linalg.norm(v2)

                # transformer vector
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                trans_ang_loss = (np.arccos(dot_product) * 180 / np.pi)  # angle in degrees
                trans_dis_loss = np.linalg.norm(np.array([target_x, target_y]) - np.array([trans_pred_x, trans_pred_y]))

                # visualization the flips
                if savefigure:
                    os.makedirs(fig_path, exist_ok=True)
                    outfig = plot_gaze_largedata(img, flips[i], [eye_x, eye_y], [target_x,target_y],\
                                        [trans_pred_x, trans_pred_y])
                    plt.text(.5*w, 1.2*h, 'transformer ang_error:{:.2f}, eucli_error:{:.2f}'.format(trans_ang_loss,trans_dis_loss), \
                             horizontalalignment='center',
                             verticalalignment='bottom')
                    plt.savefig('{}/result_{}'.format(fig_path, images_name[i].split('/')[-1]))
                    plt.close()

                # CALCULATE ERROR FOR EACH IMAGE
                TRANS_PRED.append([trans_pred_x, trans_pred_y])
                TRANS_ANG_LOSS.append(trans_ang_loss)
                TRANS_DIS_LOSS.append(trans_dis_loss)
                GAZE_START.append([eye_x, eye_y])
                GT_GAZE.append([target_x, target_y])

            # FOR EACH BATCH
            IMAGES += list(images_name)
            print(len(IMAGES))

        # save data in the original image coordination
        output = pd.DataFrame({'image': IMAGES,
                               'gaze_start_x': np.array(GAZE_START)[:, 0],
                               'gaze_start_y': np.array(GAZE_START)[:, 1],
                               'gazed_x': np.array(GT_GAZE)[:, 0],
                               'gazed_y': np.array(GT_GAZE)[:, 1],
                               'transformer_est_x': np.array(TRANS_PRED)[:, 0],
                               'transformer_est_y': np.array(TRANS_PRED)[:, 1],
                               'transformer_ang_error': np.array(TRANS_ANG_LOSS),
                               'transformer_eucli_error': np.array(TRANS_DIS_LOSS),
                               })
    return output


def evaluate_test(anno_path, test_img_path, test_bbx_path, chong_est, criterion, model, fig_path, savefigure=True):
    '''

    @param anno_path:gazed location
    @type anno_path: str
    @param test_img_path:test image path
    @type test_img_path: str
    @param test_bbx_path:test bounding box path
    @type test_bbx_path: str
    @param chong_est: chong estimation on test images
    @param criterion: loss criterion
    @param model: the model
    @return:output
    @rtype:dataframe
    '''

    IMAGES = []
    GAZE_START = []
    GT_GAZE = []

    TRANS_PRED = []
    TRANS_ANG_LOSS = []
    TRANS_DIS_LOSS = []
    CHONG_PRED = []
    CHONG_ANG_LOSS = []
    CHONG_DIS_LOSS = []

    test_data = GazeDataloader(anno_path, test_img_path, test_bbx_path)
    test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True)
    test_dataiter = iter(test_dataloader)
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for images_name, images, flips, h_crops, masks, eye, targetgaze, displacexy in test_dataiter:
            eye, displacexy = np.array(eye), np.array(displacexy)
            targetgaze_label = np.array(targetgaze['labels'].tolist())
            targetgaze_bbx = np.array(targetgaze['boxes']) #x_cen, y_cen, width, height

            test_b_size = images.shape[0]
            gaze_pred = model(images, h_crops, masks)
            gaze_pred_logits = np.array(gaze_pred['pred_logits'].detach()) # bs x 100 x 2
            gaze_pred_bbx = np.array(gaze_pred['pred_boxes'].detach())  # bs x 100 x 4

            # loss
            targets = [{'labels': targetgaze['labels'][i][0].unsqueeze(0).to(device),
                        'boxes': targetgaze['boxes'][i].unsqueeze(0).to(device)} \
                       for i in range(test_b_size)]
            indices = np.array(criterion.matcher(gaze_pred, targets))
            for i in range(test_b_size):
                img = plt.imread(images_name[i])
                try: h, w, _ = img.shape
                except:h, w = img.shape

                # find the bbox idx out of 100 queries
                idx = indices[i][0]
                gaze_bbx = gaze_pred_bbx[i][idx]

                # chong prediction (relative within the image)
                chong = chong_est[chong_est['frame'].str.contains(images_name[i].split('/')[-1])]
                chong_pred_x, chong_pred_y = chong['x_r'].item(), chong['y_r'].item() #prediction in the original image

                # transformer prediction (relative with background)
                disx, disy = displacexy[i]
                # # transform gaze_pred, targetgaze, eye position from background to image
                # trans_pred_x, trans_pred_y = coord_bg2img(gaze_bbx[0], gaze_bbx[1], disx, disy)
                # eye_x, eye_y = coord_bg2img(eye[i][0], eye[i][1], disx, disy)
                # target_x, target_y = coord_bg2img(targetgaze_bbx[i][0], targetgaze_bbx[i][1], disx, disy)
                trans_pred_x, trans_pred_y = gaze_bbx[0], gaze_bbx[1]
                eye_x, eye_y = eye[i][0], eye[i][1]
                target_x, target_y = targetgaze_bbx[i][0], targetgaze_bbx[i][1]
                # flip transformer x if the image is flipped horizontally, keep all xy in original image coordination
                if flips[i]:
                    trans_pred_x = 1- trans_pred_x
                    eye_x = 1 - eye_x
                    target_x = 1-target_x

                vec1 = (target_x - eye_x, target_y - eye_y) #groundtruth vector
                vec2 = (trans_pred_x - eye_x, trans_pred_y - eye_y) # transformer estimation
                vec3 = (chong_pred_x - eye_x, chong_pred_y - eye_y) # chong estimation

                v1, v2, v3 = [vec1[0] * w, vec1[1] * h], \
                         [vec2[0] * w, vec2[1] * h], \
                        [vec3[0] * w, vec3[1] * h]
                unit_vector_1 = v1 / np.linalg.norm(v1)
                unit_vector_2 = v2 / np.linalg.norm(v2)
                unit_vector_3 = v3/ np.linalg.norm(v3)

                # transformer vector
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                trans_ang_loss = (np.arccos(dot_product) * 180 / np.pi)  # angle in degrees
                trans_dis_loss = np.linalg.norm(np.array([target_x, target_y]) - np.array([trans_pred_x, trans_pred_y]))

                # chong vector
                dot_product = np.dot(unit_vector_1, unit_vector_3)
                chong_ang_loss = (np.arccos(dot_product) * 180 / np.pi)  # angle in degrees
                chong_dis_loss = np.linalg.norm(np.array([target_x, target_y]) - np.array([chong_pred_x, chong_pred_y]))

                # visualization the flips
                if savefigure:
                    os.makedirs(fig_path, exist_ok=True)
                    outfig = plot_gaze_largedata(img, flips[i], [eye_x, eye_y], [target_x,target_y],\
                                        [trans_pred_x, trans_pred_y], [chong_pred_x, chong_pred_y] )
                    plt.text(.5*w, 1.2*h, 'transformer ang_error:{:.2f}, eucli_error:{:.2f}'.format(trans_ang_loss,trans_dis_loss), \
                             horizontalalignment='center',
                             verticalalignment='bottom')
                    plt.text(.5*w, 1.25*h ,'chong ang_error:{:.2f}, eucli_error:{:.2f}'.format(chong_ang_loss, chong_dis_loss), \
                             horizontalalignment='center',
                             verticalalignment='bottom')

                    plt.savefig('{}/result_{}'.format(fig_path, images_name[i].split('/')[-1]))
                    plt.close('all')


                # CALCULATE ERROR FOR EACH IMAGE
                TRANS_PRED.append([trans_pred_x, trans_pred_y])
                TRANS_ANG_LOSS.append(trans_ang_loss)
                TRANS_DIS_LOSS.append(trans_dis_loss)
                CHONG_PRED.append([chong_pred_x, chong_pred_y])
                CHONG_ANG_LOSS.append(chong_ang_loss)
                CHONG_DIS_LOSS.append(chong_dis_loss)
                GAZE_START.append([eye_x, eye_y])
                GT_GAZE.append([target_x, target_y])


            # FOR EACH BATCH
            IMAGES += list(images_name)
            print(len(IMAGES))

        # save data in the original image coordination
        output = pd.DataFrame({'image': IMAGES,
                               'gaze_start_x': np.array(GAZE_START)[:, 0],
                               'gaze_start_y': np.array(GAZE_START)[:, 1],
                               'gazed_x': np.array(GT_GAZE)[:, 0],
                               'gazed_y': np.array(GT_GAZE)[:, 1],
                               'transformer_est_x': np.array(TRANS_PRED)[:, 0],
                               'transformer_est_y': np.array(TRANS_PRED)[:, 1],
                               'transformer_ang_error': np.array(TRANS_ANG_LOSS),
                               'transformer_eucli_error': np.array(TRANS_DIS_LOSS),
                               'chong_est_x': np.array(CHONG_PRED)[:, 0],
                               'chong_est_y': np.array(CHONG_PRED)[:, 1],
                               'chong_ang_error': np.array(CHONG_ANG_LOSS),
                               'chong_eucli_error': np.array(CHONG_DIS_LOSS),
                               })
    return output


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
epoch=220
checkpoint = torch.load('trainedmodels/model_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
# plt.plot(checkpoint['train_loss'])
# plt.plot(checkpoint['test_loss'])
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)


# evaluate both models' estimation on viu dataset
datapath = '/Users/nicolehan/Documents/Research/gazetransformer'
fig_path='{}/model_eval_outputs/epoch{}_head_outputs'.format(datapath,epoch)
anno_path = "{}/data/annotations".format(datapath)
train_img_path = "{}/data/train_s".format(datapath)
train_bbx_path = "{}/data/train_bbox".format(datapath)
test_img_path = "{}/data/test".format(datapath)
test_bbx_path = "{}/data/test_bbox".format(datapath)
matcher = build_matcher(set_cost_class=1, set_cost_bbox=5, set_cost_giou=2)
weight_dict = {'loss_ce': 1, 'loss_bbox': 20, 'loss_giou': 2}
losses = ['labels', 'boxes']
num_classes = 1
criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                         eos_coef=0.01, losses=losses)
# eos_coef: weight to the non-gazed location for class imbalance
chong_est = pd.read_excel('{}/data/Chong_estimation_test.xlsx'.format(datapath))
# output = evaluate_train(anno_path, train_img_path, train_bbx_path, chong_est, criterion, model, fig_path, savefigure=True)
output = evaluate_test(anno_path, test_img_path, test_bbx_path, chong_est, criterion, model, fig_path, savefigure=True)
output.to_excel('{}/model_eval_outputs/transformer_headbody_epoch{}_result.xlsx'.format(datapath, epoch), index=None)
analyze_error(output, epoch, '{}/model_eval_outputs'.format(datapath))
