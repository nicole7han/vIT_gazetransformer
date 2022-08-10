import os, json, sys, torch, glob, random, cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageOps
from numpy import unravel_index
import pandas as pd
import seaborn as sns
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

sys.path.append('/Users/nicolehan/Documents/Research/vIT_gazetransformer')
from script.model import *
from script.utils import *


def sns_setup(sns, fig_size=(10, 8)):
    sns.set(rc={'figure.figsize': fig_size})
    sns.set_context("paper", rc={"font.size": 30, "axes.titlesize": 40, "axes.labelsize": 30,
                                 "legend.title_fontsize": 30, "legend.fontsize": 20,
                                 "xtick.labelsize": 30, "ytick.labelsize": 30,
                                 "figure.titlesize": 20,
                                 })

    sns.set_style("white")
    sns.set_palette("deep")
    return


def sns_setup_wide(sns):
    sns.set(rc={'figure.figsize': (12, 7)})
    sns.set_context("paper", rc={"font.size": 35, "axes.titlesize": 40, "axes.labelsize": 30,
                                 "legend.title_fontsize": 30, "legend.fontsize": 30,
                                 "xtick.labelsize": 30, "ytick.labelsize": 30,
                                 "legend.frameon": False, "figure.titlesize": 20,
                                 })

    sns.set_style("white")
    sns.set_palette("Set2")
    return


def sns_setup_small(sns, fig_size=(12, 8)):
    sns.set(rc={'figure.figsize': fig_size})
    sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20,
                                 "legend.title_fontsize": 20, "legend.fontsize": 15,
                                 "xtick.labelsize": 20, "ytick.labelsize": 20})
    sns.set_style("white")
    sns.set_palette("Set2")
    return


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
def transform(x):
    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    x = trans(x)
    return x


def visualize_result(image, gaze_map, out_map, idx=0):
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].imshow(np.transpose(image[idx, ::], (1, 2, 0)))
    axs[0].title.set_text('original image')
    axs[1].imshow(gaze_map[idx, 0, ::])
    axs[1].title.set_text('groundtruth gaze map')
    axs[2].imshow(out_map[idx, 0, ::])
    axs[2].title.set_text('prediction gaze map')


def plot_gaze(h_yxhw, b_yxhw, image, gaze_map, out_map, chong_model_est=None, idx=0):
    '''
    :param h_yxhw: head bounding box (y, x, h, w)
    :param b_yxhw: body bounding box (y, x, h, w)
    :param image: images (b_size x 3 x 256 x 256)
    :param gaze_map: ground truth gaze map (b_size x 3 x 256 x 256)
    :param out_map: gaze estimation map from transformer (b_size x 3 x 256 x 256)
    :param chong_model_est: gaze estimation from chong model
    :param idx: index of image to visualize
    :return:
        (gaze_s_y, gaze_s_x): gaze start (y,x),
        pred_gaze: prediction gaze estimation (y,x)
        gt_gaze: groundtruth gaze location (y,x)
    '''

    h, w = image.shape[2], image.shape[3]
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    img = np.transpose(image[idx, ::], (1, 2, 0)).copy()

    # transformer gaze estimation (blue)
    out_map_s = gaussian_filter(out_map[idx, 0, ::], 3)
    pred_gaze = np.array(unravel_index(out_map_s.argmax(), out_map_s.shape)) / out_map_s.shape[0]
    gaze_s_y, gaze_s_x, gaze_e_y, gaze_e_x = int((h_yxhw[0] + .5 * h_yxhw[2]) * h), \
                                             int((h_yxhw[1] + .5 * h_yxhw[3]) * w), \
                                             int((pred_gaze[0]) * h), \
                                             int((pred_gaze[1]) * w)
    img = cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_e_x, gaze_e_y), (0, 0, 1), 2)  #

    # chong gaze estimation (yellow)
    try:
        chong_x, chong_y = int(chong_model_est[0] * w), int(chong_model_est[1] * h)
        img = cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (chong_x, chong_y), (1, .72, .05), 2)
    except: pass
    # groundtruth gaze (green)
    gt_gaze = np.array(unravel_index(gaze_map[idx, 0, ::].argmax(), gaze_map[idx, 0, ::].shape)) / \
              gaze_map[idx, 0, ::].shape[0]
    gaze_y, gaze_x = int((gt_gaze[0]) * w), int((gt_gaze[1]) * w)
    img = cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_x, gaze_y), (0, 1, 0), 2)
    axs[0].imshow(img)

    # bounding box
    h_rect = patches.Rectangle((h_yxhw[1] * w, h_yxhw[0] * h), h_yxhw[3] * w, h_yxhw[2] * h, linewidth=1, edgecolor='r',
                               facecolor='none')
    # b_rect = patches.Rectangle((b_yxhw[1] * w, b_yxhw[0] * h), b_yxhw[3] * w, b_yxhw[2] * h, linewidth=1, edgecolor='r',
    #                            facecolor='none')
    axs[0].add_patch(h_rect)
    # axs[0].add_patch(b_rect)
    axs[0].title.set_text('original image')

    # heatmap
    axs[1].imshow(gaze_map[idx, 0, ::])
    axs[1].title.set_text('groundtruth gaze map')
    axs[2].imshow(out_map_s)
    axs[2].title.set_text('prediction gaze map')

    return np.array([gaze_s_y / h, gaze_s_x / w]), pred_gaze, gt_gaze


class GazeDataloader_gazevideo(Dataset):
    """
    Dataloader class
    Returns:
        img: original image,
        h_crop: cropped head region,
        b_crop: cropped body region,
        g_crop: cropped gaze region,
        masks: binary masks of head, body, gaze region
        gaze map: resized gaze binary map
        img_anno: eye and gaze annotation locations
    """

    def __init__(self, anno_path, img_path, bbx_path, gazer_bbox):
        # PARAMETERS:
        # anno_path: gazed location
        # img_path: train/test images
        # bbx_path: train/test head and body bounding box
        # bbox_region: 'hb', 'h', or 'b'
        # RETURNS:
        # image, gaze_map

        self.anno_path = anno_path
        self.eye_gaze = pd.read_excel(self.anno_path, engine='openpyxl')
        self.img_path = img_path
        self.bbx_path = bbx_path
        self.img_paths = os.listdir(self.img_path)
        self.img_paths = [f for f in self.img_paths if not f.startswith('.')]
        self.mode = 'train' if 'train' in self.img_path.split("/")[-1] else 'test'

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            #                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.resize = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])

    def __len__(self):
        num_imgs = len(self.img_paths)
        return num_imgs

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        inputs = plt.imread('{}/{}'.format(self.img_path, img_name))
        image = Image.open('{}/{}'.format(self.img_path, img_name))
        w, h = image.size
        img = self.transform(image)

        # load groundtruth gaze location
        orig_img_name = img_name.split('_')[0]
        g_x, g_y = self.eye_gaze[self.eye_gaze['Video'] == orig_img_name]['gazed_locationx'].tolist()[0] / 800, \
                   self.eye_gaze[self.eye_gaze['Video'] == orig_img_name]['gazed_locationy'].tolist()[0] / 600

        # # load bbox mask
        # with open('{}/{}.json'.format(self.bbx_path, img_name.split('.jpg')[0])) as file:
        #     img_bbx = json.load(file)
        # num_gazer = int(len(img_bbx)/8)
        # for g in range(num_gazer):
        #     if gazer_bbox!='hb': # topleft, width
        #         bbx_x, bbx_y, bbx_w, bbx_h = img_bbx['{}_x{}'.format(gazer_bbox,g)],img_bbx['{}_y{}'.format(gazer_bbox,g)], \
        #                              img_bbx['{}_w{}'.format(gazer_bbox, g)],img_bbx['{}_h{}'.format(gazer_bbox,g)]
        #     else:
        #         h_x, h_y, h_w, h_h, b_x, b_y, b_w, b_h = \
        #             img_bbx['h_x{}'.format(g)],img_bbx['h_y{}'.format(g)],img_bbx['h_w{}'.format(g)],img_bbx['h_h{}'.format(g)], \
        #             img_bbx['b_x{}'.format(g)], img_bbx['b_y{}'.format(g)], img_bbx['b_w{}'.format(g)], img_bbx[
        #                 'b_h{}'.format(g)]
        #         bbx_x, bbx_y, bbx_w, bbx_h = min(h_x,b_x),min(h_y, b_y),  max(h_w, b_w), max(h_h,b_h)
        #
        #     bbx_crop = inputs[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)]
        #     bbx_crop = self.transform(Image.fromarray(bbx_crop))
        #     mask = np.zeros([h, w])
        #     mask[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)] = 1
        #     mask = Image.fromarray(mask)
        #     mask = torch.tensor(np.array(mask.resize([224, 224]))).unsqueeze(0)
        #     if g == 0:
        #         gazer_crops = bbx_crop.unsqueeze(0)
        #         gazer_masks = mask.unsqueeze(0)
        #     else:
        #         gazer_crops = torch.cat([gazer_crops, bbx_crop.unsqueeze(0)])
        #         gazer_masks = torch.cat([gazer_masks, mask.unsqueeze(0)])

        return img_name, img, {'labels':torch.tensor([1]) ,'boxes':torch.tensor([g_x,g_y])}


def plot_gaze_viudata(img, eyexy, targetxy, transxy, chongxy=None):
    # take original image and coordinations
    # flip the image and coordination if flip==True
    try:
        h, w, _ = img.shape
    except:
        h, w = img.shape

    # fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    # transformer gaze estimation (blue)
    gaze_pred_x, gaze_pred_y = int(transxy[0] * w), \
                               int(transxy[1]* h)
    try:
        chong_pred_x, chong_pred_y = int(chongxy[0] * w), \
                                     int(chongxy[1] * h)
    except:
        pass
    gaze_s_x, gaze_s_y, gaze_e_x, gaze_e_y = int(eyexy[0] * w), \
                                             int(eyexy[1] * h), \
                                             int(targetxy[0] * w), \
                                             int(targetxy[1] * h)
    # transformer prediction (blue)
    cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_pred_x, gaze_pred_y), (0, 0, 255), 2)

    # chong prediction (orange)
    try: cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (chong_pred_x, chong_pred_y), (255, 255, 0), 2)
    except: pass

    # groundtruth gaze (green)
    cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_e_x, gaze_e_y), (0, 255, 0), 2)

    return img

def evaluate_2model(anno_path, test_img_path, test_bbx_path, chong_est, model, fig_path, criterion,
                    bbx_noise=False, gazer_bbox='hb', cond='intact', savefigure=True, mode='map'):
    '''
    @param anno_path:    output = evaluate_2model(anno_path, test_img_path, test_bbx_path, None, model, fig_path, criterion, gazer_bbox=gazer_bbox) gazed location
    @param test_img_path: test image path
    @param test_bbx_path: test bounding box path
    @param chong_est: chong model estimation excel
    @param model: transformer model
    @param fig_path: figure destination path
    @param bbx_noise: add jitter to box position or not
    @param gazer_bbox: what region used to predict gaze 'h':head, 'b':body, 'hb':whole head and body region
    @return:
        output: excel sheet with euclidean error and angular error of both models (gaze transformer and chong et 2020)
    '''

    model.eval()
    test_data = GazeDataloader_gazevideo(anno_path, test_img_path, test_bbx_path, gazer_bbox)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
    test_dataiter = iter(test_dataloader)

    IMAGES = []
    GAZE_START = []
    PREDICT_GAZE = []
    CHONG_PREDICT_GAZE = []
    GT_GAZE = []
    PERSON_IDX = []

    bbx = np.load('/Users/nicolehan/Documents/Research/gazetransformer/gaze_video_data/bbx_viu_images.npy', allow_pickle=True)
    bbx = bbx[()]
    for images_name, images, targetgaze in test_dataiter:
#        print(len(IMAGES))
        print(images_name)
        test_b_size = images.shape[0]

        # load chong model estimation for the image
        try:
            chong_image_est = None
            chong_image_est = chong_est[chong_est['image']==images_name[0].replace('{}_'.format(cond),'')]
            if len(chong_image_est) == 0:
                pass
        except: pass
        # load image
        inputs = plt.imread('{}/{}'.format(test_img_path, images_name[0]))
        h, w, _ = inputs.shape

        try:
            headbody = bbx[
                '/Users/nicolehan/Documents/Research/gazetransformer/gaze_video_data/transformer_all_img/{}'.format(
                    images_name[0].replace('{}_'.format(cond),''))]
        except:
            continue
        num_people = int(len(headbody) / 2)

        # for each image, loop through all the gaze-orienting individual
        for p in range(num_people):
            try:
                h_y, h_x, h_h, h_w = headbody['head{}'.format(p)]
                b_y, b_x, b_h, b_w = headbody['body{}'.format(p)]
                hb_y, hb_x, hb_h, hb_w = min(h_y,b_y), min(h_x,b_x), max(h_h,b_h), max(h_w,b_w)
            except:
                continue

            # model trained on heads, only look at head region when head is available
            # if cond != 'nh':
            #     bbx_y, bbx_x, bbx_h, bbx_w = h_y, h_x, h_h, h_w
            # else:
            #     bbx_y, bbx_x, bbx_h, bbx_w = b_y, b_x, b_h, b_w
            bbx_y, bbx_x, bbx_h, bbx_w = h_y, h_x, h_h, h_w

            # load head and body masks + crops
            masks = torch.zeros([224, 224])
            masks[int(bbx_y * 224):int((bbx_y + bbx_h) * 224), int(bbx_x * 224):int((bbx_x + bbx_w) * 224)] = 1
            masks = masks.unsqueeze(0).unsqueeze(0)
            box_crops = inputs[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)]
            box_crops = transform(Image.fromarray(box_crops)).unsqueeze(0)

            # find corresponding gaze-orienting in the chong model estimation
            try:
                cenx, ceny = h_x+.5*h_w, h_y+.5*h_h
                chong_model_est = chong_image_est[abs(chong_image_est['gaze_start_x']-cenx)<0.001]
                if len(chong_model_est)>1:
                    print(images_name[0])
                    print('not unique!!')
                    continue
                chong_model_est = np.array([chong_model_est['chong_est_x'].item(), chong_model_est['chong_est_y'].item()])

            except: pass

            gaze_pred = model(images, box_crops, masks)
            gaze_pred_logits = np.array(gaze_pred['pred_logits'].detach())[0] # 100 x 2
            gaze_pred_prob = np.array(gaze_pred["pred_logits"].flatten(0, 1).softmax(-1).detach())
            gaze_pred_bbx = np.array(gaze_pred['pred_boxes'].detach())[0]  # 100 x 4
            idx = gaze_pred_prob[:, 1].argmax() # get maximum logit prediction for gazed location

#            # loss
#            targets = [{'labels': targetgaze['labels'][i][0].unsqueeze(0).to(device),
#                        'boxes': targetgaze['boxes'][i].unsqueeze(0).to(device)} \
#                       for i in range(test_b_size)]
#            indices = np.array(criterion.matcher(gaze_pred, targets))
#            idx = indices[0][0]


            # result
            transxy = gaze_pred_bbx[idx]
            eyexy = np.array([h_x+0.5*h_w, h_y+0.5*h_h])
            targetxy = np.array(targetgaze['boxes'][0])

            # visualization
            os.makedirs(fig_path, exist_ok=True)
            if savefigure:
                plt.close()
                rect = patches.Rectangle((int(bbx_x * w), int((bbx_y) * h)),
                                         int(bbx_w * w), int(bbx_h * h), linewidth=2, edgecolor=(0, 1, 0),
                                         facecolor='none')
                if mode == 'arrow':
                    fig = plt.figure()
                    plt.axis('off')
                    ax = plt.gca()
                    img = plt.imread('{}/{}'.format(test_img_path, images_name[0]))
                    img = plot_gaze_viudata(img, eyexy, targetxy, transxy)
                    plt.imshow(img)
                    ax.add_patch(rect)
                    ax.set_axis_off()
                    fig.savefig('{}/{}_person{}_arrow.jpg'.format(fig_path, images_name[0], p + 1))
                    plt.close()
                elif mode == 'map':
                    heatmap = np.zeros([h, w])
                    for i in range(len(gaze_pred_logits)):
                        prob = gaze_pred_prob[i][1]
                        locx, locy = gaze_pred_bbx[i]
                        locx, locy = int(locx * w), int(locy * h)
                        heatmap[locy - int(.02 * w):locy + int(.02 * w), locx - int(.02 * h):locx + int(.02 * h)] = prob
                    heatmap = heatmap / (heatmap.max())
                    heatmap = gaussian_filter(heatmap, 40)
                    fig, ax = plt.subplots()
                    img = plt.imread('{}/{}'.format(test_img_path, images_name[0]))
                    gaze_s_x, gaze_s_y, gaze_e_x, gaze_e_y = int(eyexy[0] * w), \
                                         int(eyexy[1] * h), \
                                         int(targetxy[0] * w), \
                                         int(targetxy[1] * h)
                    # groundtruth gaze (green)
                    cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_e_x, gaze_e_y), (0, 255, 0), 2)
                    ax.imshow(img)
                    ax.imshow(heatmap, alpha=.4)
                    ax.set_axis_off()
                    fig.savefig('{}/{}_person{}_map.jpg'.format(fig_path, images_name[0], p + 1))
                    plt.close()

            IMAGES.append(images_name[0])
            GAZE_START.append(eyexy)
            PREDICT_GAZE.append(transxy)
            GT_GAZE.append(targetxy)
            PERSON_IDX.append(p + 1)
            try: CHONG_PREDICT_GAZE.append(chong_model_est)
            except: pass

    try:
        output = pd.DataFrame({'image': IMAGES,
                               'gazer': PERSON_IDX,
                               'gaze_start_x': np.array(GAZE_START)[:, 0],
                               'gaze_start_y': np.array(GAZE_START)[:, 1],
                               'gazed_x': np.array(GT_GAZE)[:, 0],
                               'gazed_y': np.array(GT_GAZE)[:, 1],
                               'transformer_est_x': np.array(PREDICT_GAZE)[:, 0],
                               'transformer_est_y': np.array(PREDICT_GAZE)[:, 1],
                               'chong_est_x': np.array(CHONG_PREDICT_GAZE)[:, 0],
                               'chong_est_y': np.array(CHONG_PREDICT_GAZE)[:, 1],
                               })
    except:
        output = pd.DataFrame({'image': IMAGES,
                               'gazer': PERSON_IDX,
                               'gaze_start_x': np.array(GAZE_START)[:, 0],
                               'gaze_start_y': np.array(GAZE_START)[:, 1],
                               'gazed_x': np.array(GT_GAZE)[:, 0],
                               'gazed_y': np.array(GT_GAZE)[:, 1],
                               'transformer_est_x': np.array(PREDICT_GAZE)[:, 0],
                               'transformer_est_y': np.array(PREDICT_GAZE)[:, 1],
                               })
    return output


def coord_bg2img(x, y, disx, disy, img_size=100, bg_size=224):
    # coordinates of x, y from an image displaced on a bacground with displacement xy -> xy within the image itself
    x = x*bg_size-(disx*bg_size-img_size/2)
    y = y*bg_size-(disy*bg_size-img_size/2)
    return x/img_size, y/img_size



def plot_gaze_largedata(img, flip, eyexy, targetxy, transxy, chongxy=None):
    # take original image and coordinations
    # flip the image and coordination if flip==True
    try:
        h, w, _ = img.shape
    except:
        h, w = img.shape

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    # transformer gaze estimation (blue)
    gaze_pred_x, gaze_pred_y = int(transxy[0] * w), \
                               int(transxy[1]* h)
    try:
        chong_pred_x, chong_pred_y = int(chongxy[0] * w), \
                                     int(chongxy[1] * h)
    except:
        pass
    gaze_s_x, gaze_s_y, gaze_e_x, gaze_e_y = int(eyexy[0] * w), \
                                             int(eyexy[1] * h), \
                                             int(targetxy[0] * w), \
                                             int(targetxy[1] * h)
    if flip == True:
        image = Image.fromarray(img).transpose(Image.FLIP_LEFT_RIGHT)
        img = np.array(image)
        gaze_s_x, gaze_e_x = w-gaze_s_x, w-gaze_e_x
        try:  chong_pred_x = w-chong_pred_x
        except: pass
        gaze_pred_x = w-gaze_pred_x
        axs.title.set_text('original image (flipped)')
    else:
        axs.title.set_text('original image')

    # transformer prediction (blue)
    img = cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_pred_x, gaze_pred_y), (0, 0, 255), 2)

    # chong prediction (yellow)
    try: img = cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (chong_pred_x, chong_pred_y), (255, 255, 0), 2)
    except: pass

    # groundtruth gaze (green)
    img = cv2.arrowedLine(img, (gaze_s_x, gaze_s_y), (gaze_e_x, gaze_e_y), (0, 255, 0), 2)
    # axs.imshow(img)
    plt.imshow(img)
    # bounding box
    # h_rect = patches.Rectangle((h_yxhw[1]*w, h_yxhw[0]*h), h_yxhw[3]*w, h_yxhw[2]*h, linewidth=1, edgecolor='r', facecolor='none')
    # b_rect = patches.Rectangle((b_yxhw[1]*w, b_yxhw[0]*h), b_yxhw[3]*w, b_yxhw[2]*h, linewidth=1, edgecolor='r', facecolor='none')
    # axs[0].add_patch(h_rect)
    # axs[0].add_patch(b_rect)

    # # heatmap
    # gaze_map = gaussian_filter(gaze_map, 3)
    # axs[1].imshow(gaze_map)
    # axs[1].title.set_text('groundtruth gaze map')


def organize_gazedata():
    headbody_path = 'gaze_video_data/boundingbox_gaze-orienting people head_body'
    img_path = '/Users/nicolehan/Documents/Research/GazeExperiment/Mechanical turk/Mechanical Turk (target absent)/Mechanical Turk_orig/done'
    img_dest_path = '/Users/nicolehan/Documents/Research/GazeExperiment/Mechanical turk/Mechanical Turk (target absent)/Mechanical Turk_orig'

    bbx_list = os.listdir(headbody_path)
    bbx_list = [f for f in bbx_list if not f.startswith('.')]
    bbx_list.sort()
    bbx_folder_nums = len(bbx_list)

    x_train, x_test = train_test_split(bbx_list, test_size=0.2)
    for img in x_train:
        src = '{}/{}'.format(headbody_path, img)
        dest = '{}/transformer_train_bbx/{}'.format(img_dest_path, img)
        shutil.copy(src, dest)
        img_name = img.split('.json')[0] + '.jpg'
        src = '{}/{}'.format(img_path, img_name)
        dest = '{}/transformer_train_img/{}'.format(img_dest_path, img_name)
        shutil.copy(src, dest)

    for img in x_test:
        src = '{}/{}'.format(headbody_path, img)
        dest = '{}/transformer_test_bbx/{}'.format(img_dest_path, img)
        shutil.copy(src, dest)
        img_name = img.split('.json')[0] + '.jpg'
        src = '{}/{}'.format(img_path, img_name)
        dest = '{}/transformer_test_img/{}'.format(img_dest_path, img_name)
        shutil.copy(src, dest)
