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


def visualize_result(images_name, flips, g_crops, gaze_maps, out_map, idx=0):
    if idx is None:
        idx = np.random.randint(0, len(images_name))
    img_name = images_name[idx]
    img = Image.fromarray(plt.imread(img_name)).resize((224, 224))

    gaze_maps = gaussian_smooth(gaze_maps, 21, 5)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if flips[idx]:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        axs[0, 0].title.set_text('original image (flipped)')
    else:
        axs[0, 0].title.set_text('original image')
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(np.transpose(g_crops[idx, :, :, :].detach().cpu().numpy(), (1, 2, 0)))
    axs[0, 1].title.set_text('gazed image')
    axs[1, 0].imshow(gaze_maps[idx, 0, :, :].detach().cpu().numpy(), "gray")
    axs[1, 0].title.set_text('gazed mask (ground truth)')
    axs[1, 1].imshow(out_map[idx, 0, :, :].detach().cpu().numpy(), "gray")
    axs[1, 1].title.set_text('gazed mask (prediction)')


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
        self.mode = self.img_path.split("/")[-1]

        img_folder_list = os.listdir(self.img_path)
        img_folder_list = [f for f in img_folder_list if not f.startswith('.')]
        img_folder_list.sort()
        img_folder_nums = len(img_folder_list)
        for i in range(img_folder_nums):  # each image folder
            img_list = glob.glob('{}/{}/*'.format(self.img_path, img_folder_list[i]))
            img_list.sort()
            if len(img_list) > 0:
                self.img_paths += img_list

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
        # config = resolve_data_config({}, model=self.vit)
        # transform = create_transform(**config)
        # inputs = transform(img).unsqueeze(0)
        # vit_feature_extractor = torch.nn.Sequential(*list(self.vit.children())[:-1])
        # vit_feature = vit_feature_extractor(inputs)  # input size 3x384x384

        # name = 'data/test/00000000/00000349.jpg'
        img_name = name.split("/")[-1]  # name = 'data/train/00000004/00004947.jpg'
        folder_name = name.split("/")[-2]
        inputs = plt.imread(name)
        try:
            h, w, _ = inputs.shape
        except:
            h, w = inputs.shape
            print('image {} has only one channel'.format(img_name))
            inputs = np.stack([inputs, inputs, inputs], axis=-1)

        # load eye gaze annotation
        with open('{}/{}_annotation.json'.format(self.ann_path, self.mode)) as file:
            eyegaze = json.load(file)
        img_anno = eyegaze['/'.join(name.split('/')[-3:])]

        # load head and body region images
        seg_bbx = np.load("{}/bbx_{}.npy".format(self.bbx_path, folder_name), allow_pickle=True)
        seg_bbx = seg_bbx[()]

        try:
            # crop head and body 
            inputs_bbx = seg_bbx["./gazefollow/{}".format('/'.join(name.split('/')[-3:]))]

            [h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w] = inputs_bbx['head'] + inputs_bbx['body']
            h_y += random.uniform(-.05, 0)
            h_x += random.uniform(-.05, 0)
            h_h += random.uniform(0, 0.05)
            h_w += random.uniform(0, 0.05)
            b_y += random.uniform(-.05, 0)
            b_x += random.uniform(-.05, 0)
            b_h += random.uniform(0, 0.05)
            b_w += random.uniform(0, 0.05)
            # make sure all between [0,1]
            vals = [h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w]
            for i in range(len(vals)):
                if vals[i] < 0:
                    vals[i] = 0
                elif vals[i] > 1:
                    vals[i] = 1
            [h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w] = vals

            h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
            b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]

            # crop gaze direction location, default .1
            g_x, g_y = img_anno['gaze_x'], img_anno['gaze_y']
            x_l, x_h, y_l, y_h = max(0, g_x - .1), min(1, g_x + .1), max(0, g_y - .1), min(1, g_y + .1)
            g_crop = inputs[int(y_l * h):int(y_h * h), int(x_l * w):int(x_h * w)]

            # print('transform images start {}'.format(img_name))
            # create gaze map of size 64x64
            gaze_map = torch.zeros([224, 224])
            gaze_map[int(y_l * 224):int(y_h * 224), int(x_l * 224):int(x_h * 224)] = 1
            gaze_map = gaze_map.numpy()

            flip = random.random() > .5  # random flip images and corresponding
            if flip:
                # print('{} random flip'.format(img_name))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                h_crop = np.fliplr(h_crop)
                b_crop = np.fliplr(b_crop)
                g_crop = np.fliplr(g_crop)
                gaze_map = np.fliplr(gaze_map)
                # create binary masks for head, body, gaze location
                masks = torch.zeros([2, 224, 224])
                masks[0, ::][int(h_y * 224):int((h_y + h_h) * 224), int((1 - h_x - h_w) * 224):int((1 - h_x) * 224)] = 1
                masks[1, ::][int(b_y * 224):int((b_y + b_h) * 224), int((1 - b_x - b_w) * 224):int((1 - b_x) * 224)] = 1
                img_anno['eye_x'] = 1 - img_anno['eye_x']
                img_anno['gaze_x'] = 1 - img_anno['gaze_x']
            else:
                # create binary masks for head, body, gaze location
                masks = torch.zeros([2, 224, 224])  # head, body, gaze location
                masks[0, ::][int(h_y * 224):int((h_y + h_h) * 224), int(h_x * 224):int((h_x + h_w) * 224)] = 1
                masks[1, ::][int(b_y * 224):int((b_y + b_h) * 224), int(b_x * 224):int((b_x + b_w) * 224)] = 1

            gaze_map = self.resize(Image.fromarray(gaze_map))
            h_crop = self.transform(Image.fromarray(h_crop))
            b_crop = self.transform(Image.fromarray(b_crop))
            g_crop = self.transform(Image.fromarray(g_crop))
            img = self.transform(img)

        except:
            print(name)
            print('{} no data'.format(img_name))
            return


        return name, img, flip, h_crop, b_crop, g_crop, masks, gaze_map, img_anno
