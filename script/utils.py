import os, glob, json, cv2, torch, shutil, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchgeometry as tgm
import matplotlib.patches as patches
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import scipy.ndimage as ndimage
from scipy.special import softmax
from typing import Optional, List
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
import torch.distributed as dist

def str2ASCII(s):
    x = []
    for i in range(len(s)):
        x.append(ord(s[i]))
    return x


def ASCII2str(s):
    return ''.join(map(chr, s))


def resize(x, size):
    transform = transforms.Compose([
        transforms.Resize([size[0], size[1]]),
        transforms.ToTensor(),
    ])
    return transform(x)


def compute_angle(row):
    vector_1 = [row['gazed_x'] - row['gaze_start_x'], row['gazed_y'] - row['gaze_start_y']]
    vector_2 = [row['est_x'] - row['gaze_start_x'], row['est_y'] - row['gaze_start_y']]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) * 180 / np.pi  # angle in degrees
    return angle

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        # print('name: {}, required_grad: {}'.format(n, p.requires_grad))
        if(p.requires_grad) and ("bias" not in n):
            try:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                print('layer with grad:{}'.format(n))
            except:
                print('layer no grad:{}'.format(n))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def visualize_dataset(images_name, images, h_crops, b_crops, g_crops, img_annos, idx=None):
    if idx is None:
        idx = np.random.randint(0, len(images_name))
    img_n1 = images_name[idx]
    img1 = np.transpose(images[idx, :, :, :].detach().cpu().numpy(), (1, 2, 0)).copy()
    h1 = np.transpose(h_crops[idx, :, :, :].detach().cpu().numpy(), (1, 2, 0)).copy()
    b1 = np.transpose(b_crops[idx, :, :, :].detach().cpu().numpy(), (1, 2, 0)).copy()
    g1 = np.transpose(g_crops[idx, :, :, :].detach().cpu().numpy(), (1, 2, 0)).copy()
    ex, ey, gx, gy = img_annos['eye_x'][idx], img_annos['eye_y'][idx], img_annos['gaze_x'][idx], img_annos['gaze_y'][
        idx]
    img1 = cv2.arrowedLine(img1, (int(ex * 224), int(ey * 224)), (int(gx * 224), int(gy * 224)), color=[255, 255, 255],
                           thickness=3)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(img_n1)
    axs[0, 0].imshow(img1)
    axs[0, 0].title.set_text('original image')
    axs[0, 1].imshow(h1)
    axs[0, 1].title.set_text('head region')
    axs[1, 0].imshow(b1)
    axs[1, 0].title.set_text('body region')
    axs[1, 1].imshow(g1)
    axs[1, 1].title.set_text('gazed region')


def visualize_result(images_name, flips, g_crops, gaze_maps, gaze_pred, idx=0):
    if idx is None:
        idx = np.random.randint(0, len(images_name))
    img_name = images_name[idx]
    img = Image.fromarray(plt.imread(img_name)).resize((224, 224))


    gaze_maps = gaze_maps[idx, 0, ::].detach().cpu().numpy()
    gaze_maps = ndimage.gaussian_filter(gaze_maps, sigma=(5, 5))
    gaze_pred_maps = torch.zeros(gaze_maps.shape)
    gaze_pred = gaze_pred[idx].detach().numpy()
    h, w = gaze_pred_maps.shape
    gaze_pred_maps[int(h*gaze_pred[1]-2):int(h*gaze_pred[1]+2),
                    int(w*gaze_pred[0]-2):int(w*gaze_pred[0]+2)] = 1
    gaze_pred_maps = ndimage.gaussian_filter(gaze_pred_maps, sigma=(5, 5))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if flips[idx]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        axs[0, 0].title.set_text('original image (flipped)')
    else:
        axs[0, 0].title.set_text('original image')
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(np.transpose(g_crops[idx, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
    axs[0, 1].title.set_text('gazed image')
    axs[1, 0].imshow(gaze_maps, "gray")
    axs[1, 0].title.set_text('gazed mask (ground truth)')
    axs[1, 1].imshow(gaze_pred_maps, "gray")
    axs[1, 1].title.set_text('gazed mask (prediction)')



def visualize_vitpatch(img_vit_feature,spatial_attn, idx=0):
    # img_vit_feature [b_size, 14x14, 768]
    # spatial_attn [b_size, 1, 768]
    fig = plt.figure(figsize=(12, 12))
    for i in range(14): #row
        for j in range(14): # column
            patch = img_vit_feature[idx, i*14+j,:]
            patch = patch.reshape((24, 32)).detach().cpu().numpy()
            ax = fig.add_subplot(14, 14, i*14+j+1)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(patch)

    feature_attn = img_vit_feature * spatial_attn
    fig = plt.figure(figsize=(12, 12))
    for i in range(14): #row
        for j in range(14): # column
            patch = feature_attn[idx, i*14+j,:]
            patch = patch.reshape((24, 32)).detach().cpu().numpy()
            ax = fig.add_subplot(14, 14, i*14+j+1)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(patch)



def plot_gaze(h_yxhw, b_yxhw, gaze_start, pred_gaze, gt_gaze, image, gaze_map, out_map_s, idx=0):
    '''
    :param h_yxhw: head bounding box (y, x, h, w)
    :param b_yxhw: body bounding box (y, x, h, w)
    :param image: images (b_size x 3 x 256 x 256)
    :param gaze_start: the starting point of gaze
    :param pred_gaze: the gaze estimation location
    :param gt_gaze: the groundtruth gaze estimation location
    :param gaze_map: ground truth gaze map (b_size x 3 x 256 x 256)
    :param out_map_s: gaze estimation map (256 x 256)
    :param idx: index of image to visualize
    :return:
        (gaze_s_y, gaze_s_x): gaze start (y,x),
        pred_gaze: prediction gaze estimation (y,x)
        gt_gaze: groundtruth gaze location (y,x)
    '''

    h, w = image.shape[2], image.shape[3]
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    img = np.transpose(image[idx, ::], (1, 2, 0)).copy()

    # gaze estimation
    img = cv2.arrowedLine(img, (int(gaze_start[1] * h), int(gaze_start[0] * w)),
                          (int(pred_gaze[1] * h), int(pred_gaze[0] * w)), (0, 0, 255), 2)
    img = cv2.arrowedLine(img, (int(gaze_start[1] * h), int(gaze_start[0] * w)),
                          (int(gt_gaze[1] * h), int(gt_gaze[0] * w)), (0, 255, 0), 2)
    axs[0].imshow(img)

    # bounding box
    h_rect = patches.Rectangle((h_yxhw[1] * w, h_yxhw[0] * h), h_yxhw[3] * w, h_yxhw[2] * h, linewidth=1, edgecolor='r',
                               facecolor='none')
    b_rect = patches.Rectangle((b_yxhw[1] * w, b_yxhw[0] * h), b_yxhw[3] * w, b_yxhw[2] * h, linewidth=1, edgecolor='r',
                               facecolor='none')
    axs[0].add_patch(h_rect)
    axs[0].add_patch(b_rect)
    axs[0].title.set_text('original image')

    # heatmap
    axs[1].imshow(gaze_map[idx, 0, ::])
    axs[1].title.set_text('groundtruth gaze map')
    axs[2].imshow(out_map_s)
    axs[2].title.set_text('prediction gaze map')


def gaussian_smooth(masks, k_size, sd):
    maps = tgm.image.gaussian_blur(masks, (k_size, k_size), (sd, sd))
    return maps


def cleanup_dataset(bbx_path, img_path):
    # mask_files = os.listdir(segmask_path)
    # mask_files = [f for f in mask_files if not f.startswith('.')]
    # mask_files.sort()
    # for m in mask_files:
    # person_mask = np.load('{}/{}'.format(segmask_path,m), allow_pickle=True)
    # person_mask = person_mask[()]
    mode = img_path.split('/')[-1]
    box_files = os.listdir(bbx_path)

    for m in box_files:
        person_bbx = np.load('{}/{}'.format(bbx_path, m), allow_pickle=True)
        person_bbx = person_bbx[()]

        img_folder = img_path + '/{}'.format(m.split('_')[-1].split('.')[0])
        all_imgs = os.listdir(img_folder)
        all_imgs.sort()

        for img in all_imgs:
            inputs = plt.imread('{}/{}/{}'.format(img_path, m.split('_')[-1].split('.')[0], img))
            try:
                h, w, c = inputs.shape
            except:
                print('moving {}'.format(img))
                shutil.move('{}/{}'.format(img_folder, img), \
                            '/Users/nicolehan/Documents/Research/gazetransformer/deleted/{}/{}'.format(mode, img))
            try:
                # person_mask['./gazefollow/train/{}/{}'.format(m.split('_')[-1].split('.')[0], img)] #images with no segmentation
                h_y, h_x, h_h, h_w = \
                person_bbx['./gazefollow/{}/{}/{}'.format(mode, m.split('_')[-1].split('.')[0], img)][
                    'head']  # images with no head/body detection
                b_y, b_x, b_h, b_w = \
                person_bbx['./gazefollow/{}/{}/{}'.format(mode, m.split('_')[-1].split('.')[0], img)]['body']
                h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
                b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]

                if h_crop.shape[0] == 0 or h_crop.shape[1] == 0 or b_crop.shape[0] == 0 or b_crop.shape[1] == 0:
                    print('bounding box areas=0: moving {}'.format(img))
                    shutil.move('{}/{}'.format(img_folder, img), \
                                '/home/han/PycharmProjects/gazetransformer/deleted/{}/{}'.format(mode, img))
            except:
                print('moving {}'.format(img))
                shutil.move('{}/{}'.format(img_folder, img), \
                            '/home/han/PycharmProjects/gazetransformer/deleted/{}/{}'.format(mode, img))


# class HorizontallyFlip(object):
#     def __call__(self, img, img_name):
#         flip = (random.random()>.5)
#         if flip:
#             img = img.transpose(Image.FLIP_LEFT_RIGHT)
#             return img, flip
#         else:
#             return img, flip

class GazeDataloader(Dataset):
    """
    Dataloader class
    Returns:
        images_vitfeature: images vision transformer features
        h_crop: cropped head region, 
        b_crop: cropped body region, 
        g_crop: cropped gaze region,
        masks: binary masks of head, body, gaze region
        gaze map: resized gaze binary map
        img_anno: eye and gaze annotation locations
    """

    def __init__(self, ann_path, img_path, bbx_path):
        self.ann_path = ann_path
        self.img_path = img_path
        self.bbx_path = bbx_path
        self.img_paths = []
        self.mode = 'train' if 'train' in self.img_path.split("/")[-1] else 'test'

        img_folder_list = os.listdir(self.img_path)
        img_folder_list = [f for f in img_folder_list if not f.startswith('.')]
        img_folder_list.sort()
        img_folder_nums = len(img_folder_list)
        for i in range(img_folder_nums):  # each image folder
            img_list = glob.glob('{}/{}/*'.format(self.img_path, img_folder_list[i]))
            img_list.sort()
            if len(img_list) > 0:
                self.img_paths += img_list

        # print('{}/{}_annotation.json'.format(self.ann_path, self.mode))
        with open('{}/{}_annotation.json'.format(self.ann_path, self.mode)) as file:
            self.eyegaze = json.load(file)
        # self.transform = transforms.Compose([
        #     transforms.Resize([256, 256]),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.resize = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])

    def __len__(self):
        num_imgs = len(self.img_paths)
        return num_imgs

    def __getitem__(self, idx):
        try:
            name = self.img_paths[idx]
            img = Image.open(name)
        except:
            print(self.img_paths[idx])
        inputs = np.array(img)

        # name = 'data/test/00000000/00000349.jpg'
        img_name = name.split("/")[-1]  # name = 'data/train/00000004/00004947.jpg'
        folder_name = name.split("/")[-2]
        try:
            h, w, _ = inputs.shape
        except:
            h, w = inputs.shape
            print('image {} has only one channel'.format(img_name))
            inputs = np.stack([inputs, inputs, inputs], axis=-1)

        # load eye gaze annotation
        key = self.mode +'/' + '/'.join(name.split('/')[-2:])
        # print(key)
        img_anno = self.eyegaze[key]

        # load head and body region
        # print("{}/bbx_{}.npy".format(self.bbx_path, folder_name))
        seg_bbx = np.load("{}/bbx_{}.npy".format(self.bbx_path, folder_name), allow_pickle=True)
        seg_bbx = seg_bbx[()]
        try:
            # crop head and body
            # print("./gazefollow/{}".format(key))
            inputs_bbx = seg_bbx["./gazefollow/{}".format(key)]
            [h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w] = inputs_bbx['head'] + inputs_bbx['body']
            bbx_y, bbx_x, bbx_h, bbx_w = min(h_y, b_y), min(h_x,b_x), max(h_h,b_h), max(h_w, b_w)
            # h_y += random.uniform(-.01, 0)
            # h_x += random.uniform(-.01, 0)
            # h_h += random.uniform(0, 0.01)
            # h_w += random.uniform(0, 0.01)
            # b_y += random.uniform(-.01, 0)
            # b_x += random.uniform(-.01, 0)
            # b_h += random.uniform(0, 0.01)
            # b_w += random.uniform(0, 0.01)
            # make sure all between [0,1]
            # vals = [h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w]
            # for i in range(len(vals)):
            #     if vals[i] < 0:
            #         vals[i] = 0
            #     elif vals[i] > 1:
            #         vals[i] = 1
            # [h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w] = vals

            # h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
            # b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]
            bbx_crop = inputs[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)]

            flip = random.random() > .5  # random flip images horizontally
            if flip:
                # print('{} random flip'.format(img_name))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                bbx_crop = np.fliplr(bbx_crop)
                mask = np.zeros([ h, w])
                mask[int(bbx_y * h):int((bbx_y + bbx_h) * h), int((1 - bbx_x - bbx_w) * w):int((1 - bbx_x) * w)] = 1
                img_anno['eye_x'] = 1 - img_anno['eye_x']
                img_anno['gaze_x'] = 1 - img_anno['gaze_x']
            else:
                # create binary masks for head, body, gaze location
                mask = np.zeros([h, w])  # head, body, gaze location
                mask[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)] = 1
            bbx_crop = self.transform(Image.fromarray(bbx_crop))
        except:
            # print(name)
            # print('{} no data'.format(img_name))
            return

        # create both positive and negative examples
        labels = torch.tensor([1]).unsqueeze(0)
        boxes = torch.tensor([img_anno['gaze_x'],img_anno['gaze_y']]).unsqueeze(0)
#        while len(labels) < 20: # points close to gaze xy are also labeled as 1
#            tempx, tempy = img_anno['gaze_x']+random.uniform(-0.1, 0.1), \
#                            img_anno['gaze_y']+random.uniform(-0.1, 0.1)
#            if tempx>0 and tempx<1 and tempy >0 and tempy <1:
#                labels = torch.cat([labels, torch.tensor([1]).unsqueeze(0)], dim=0)
#                boxes  = torch.cat([boxes, torch.tensor([tempx,tempy]).unsqueeze(0)], dim=0)
#        
#        while len(labels) < 100: # points further to gaze xy are also labeled as 1
#            tempx, tempy = img_anno['gaze_x']+random.uniform(-0.5, 0.5), \
#                            img_anno['gaze_y']+random.uniform(-0.5, 0.5)
#            if tempx>0 and tempx<1 and tempy >0 and tempy <1:
#                labels = torch.cat([labels, torch.tensor([0]).unsqueeze(0)], dim=0)
#                boxes  = torch.cat([boxes, torch.tensor([tempx,tempy]).unsqueeze(0)], dim=0)
        
        
        
        img = self.transform(img)
        mask = Image.fromarray(mask)
        mask = torch.tensor(np.array(mask.resize([224,224]))).unsqueeze(0)
        # mask_bg = torch.zeros([1, 224, 224])
        # mask_bg[0, rand_y - loc_min:rand_y + loc_min, rand_x - loc_min:rand_x + loc_min] = mask
        rand_x, rand_y = 0, 0
        return name, img, flip, bbx_crop, mask, \
               torch.tensor([img_anno['eye_x'],img_anno['eye_y']]),\
               {'labels':labels ,'boxes':boxes},\
               torch.tensor([rand_x/224, rand_y/224])



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)





def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2]) # most left top
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # most right bottom

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)*self.eos_coef # gazed class + no object
        empty_weight[0] = 1 # [1, 0.01]
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) #target labels
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o #update only the matched idx with label 1, else 2 (empty)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx] #get best matching boxes
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes # l1 loss

        # loss_giou = 1 - torch.diag(generalized_box_iou(
        #     box_cxcywh_to_xyxy(src_boxes),
        #     box_cxcywh_to_xyxy(target_boxes))) # center x,y,w,h -> x0,y0, x1,y1
        # losses['loss_giou'] = loss_giou.sum() / num_boxes # generalized IOU loss
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")