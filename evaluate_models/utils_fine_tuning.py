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


# from model_patches_training.train_model_imageclips import *
def change_width(ax, new_value) :
    for patch in ax.patches :
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

    def __init__(self, anno_path, img_path, bbx_path):
        # PARAMETERS:
        # anno_path: gazed location
        # img_path: train/test images
        # bbx_path: train/test head and body bounding box
        # gaze_orien_bbx_path: all gaze-orienting people head bounding box
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
        img = Image.open('{}/{}'.format(self.img_path, img_name))
        img = self.transform(img)

        # load groundtruth gaze location
        orig_img_name = img_name.split('_')[0]
        g_x, g_y = self.eye_gaze[self.eye_gaze['Video'] == orig_img_name]['gazed_locationx'].tolist()[0] / 800, \
                   self.eye_gaze[self.eye_gaze['Video'] == orig_img_name]['gazed_locationy'].tolist()[0] / 600
        x_l, x_h, y_l, y_h = max(0, g_x - .1), min(1, g_x + .1), max(0, g_y - .1), min(1, g_y + .1)
        gaze_map = torch.zeros([224, 224])
        gaze_map[int(y_l * 224):int(y_h * 224), int(x_l * 224):int(x_h * 224)] = 1
        gaze_map = gaze_map.numpy()
        gaze_map = self.resize(Image.fromarray(gaze_map))

        # # load head and body crops
        # f = open('{}/{}.json'.format(bbx_path, img_name.split('.jpg')[0]))
        # bbx = json.load(f)

        return img_name, img, gaze_map, {'gaze_x':g_x, 'gaze_y':g_y}


def train(e_start, num_e, anno_path, train_img_path, train_bbx_path, test_img_path, test_bbx_path, b_size=20):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = Gaze_Transformer()
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=.0001, betas=(.9, .999))
    checkpoint = torch.load('models/model_epoch{}.pt'.format(e_start), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.to(device)

    LOSS = []
    for e in np.arange(e_start + 1, e_start + num_e):
        model.train()
        print('Epoch:', e, 'Training')

        train_data = GazeDataloader_gazevideo(anno_path, train_img_path, train_bbx_path)
        train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=0)
        train_dataiter = iter(train_dataloader)

        loss_iter = []
        for images_name, images, gaze_maps in train_dataiter:
            opt.zero_grad()
            images, gaze_maps = images.to(device), gaze_maps.to(device)

            b_size = images.shape[0]
            h_crops, b_crops, masks = torch.zeros(images.shape), torch.zeros(images.shape), torch.zeros(
                [b_size, 2, 256, 256])
            for i in range(b_size):  # for each image, train with all gaze-orienting people present in the image
                # load image
                inputs = plt.imread('{}/{}'.format(train_img_path, images_name[i]))
                h, w, _ = inputs.shape

                # load head and body region images
                bbx_name = images_name[i].split('.jpg')[0] + '.json'
                with open('{}/{}'.format(train_bbx_path, bbx_name)) as file:
                    # print('{}/{}'.format(train_bbx_path, bbx_name))
                    headbody = json.load(file)
                num_people = int(len(headbody) / 8)
                # randomly choose one gaze-orienting people to train
                p = random.choice(np.arange(num_people))
                h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w = headbody['h_y{}'.format(p)], headbody['h_x{}'.format(p)], \
                                                         headbody['h_h{}'.format(p)], headbody['h_w{}'.format(p)], \
                                                         headbody['b_y{}'.format(p)], headbody['b_x{}'.format(p)], \
                                                         headbody['b_h{}'.format(p)], headbody['b_w{}'.format(p)]
                h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
                b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]
                h_crop = transform(Image.fromarray(h_crop))
                b_crop = transform(Image.fromarray(b_crop))

                # load head and body masks
                mask = torch.zeros([2, 256, 256])  # head, body, gaze location
                mask[0, :, :][int(h_y * 256):int((h_y + h_h) * 256), int(h_x * 256):int((h_x + h_w) * 256)] = 1
                mask[1, :, :][int(b_y * 256):int((b_y + b_h) * 256), int(b_x * 256):int((b_x + b_w) * 256)] = 1

                h_crops[i, ::], b_crops[i, ::], masks[i, ::] = h_crop, b_crop, mask

            h_crops, b_crops, masks = h_crops.to(device), b_crops.to(device), masks.to(device)

            out_map = model(images, h_crops, b_crops, masks)  # model prediction of gaze map
            gt_map = gaussian_smooth(gaze_maps, 21, 5)
            gt_map_sums = gt_map.view(b_size, 1, -1).sum(dim=2).unsqueeze(1)  # normalize sum up to 1
            gt_map = (gt_map.view(b_size, 1, -1) / gt_map_sums).view(b_size, 1, 64, 64)

            amp_factor = 1
            loss = criterion(out_map, gt_map) * amp_factor  # amplitfy factor to avoid loss underflow
            loss.backward()
            opt.step()
            loss_iter.append(loss)

        print("training loss: {:.10f}".format(torch.mean(torch.stack(loss_iter))))
        LOSS.append(torch.mean(torch.stack(loss_iter)))

        if (e) % 10 == 0:
            if os.path.isdir('finetuning_models') == False:
                os.mkdir('finetuning_models')
            finetuning_e = e - e_start
            PATH = "finetuning_models/model_epoch{}.pt".format(finetuning_e)
            torch.save({
                'epoch': finetuning_e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': LOSS,
            }, PATH)
        if e % 1 == 0:
            # check with train images
            if os.path.isdir('finetuning_outputs') == False:
                os.mkdir('finetuning_outputs')
            try:
                for i in range(5):
                    visualize_result(images.cpu().detach().numpy(), gt_map.cpu().detach().numpy(),
                                     out_map.cpu().detach().numpy(), idx=i)
                    plt.savefig('finetuning_outputs/train_epoch{}_plot{}.jpg'.format(finetuning_e, i + 1))
                    plt.close('all')
            except:
                continue

            # check with test images
            model.eval()
            test_data = GazeDataloader_gazevideo(anno_path, test_img_path, test_bbx_path)
            test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True, num_workers=0)
            test_dataiter = iter(test_dataloader)
            images_name, images, gaze_maps = test_dataiter.next()
            images = images.to(device)
            test_b_size = images.shape[0]
            h_crops, b_crops, masks = torch.zeros(images.shape), torch.zeros(images.shape), torch.zeros(
                [test_b_size, 2, 256, 256])
            for i in range(test_b_size):  # for each image, train with all gaze-orienting people present in the image
                # load image
                inputs = plt.imread('{}/{}'.format(test_img_path, images_name[i]))
                h, w, _ = inputs.shape
                # load head and body region images
                bbx_name = images_name[i].split('.jpg')[0] + '.json'
                with open('{}/{}'.format(test_bbx_path, bbx_name)) as file:
                    headbody = json.load(file)
                num_people = int(len(headbody) / 8)
                # randomly choose one gaze-orienting people to train
                p = random.choice(np.arange(num_people))
                h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w = headbody['h_y{}'.format(p)], headbody['h_x{}'.format(p)], \
                                                         headbody['h_h{}'.format(p)], headbody['h_w{}'.format(p)], \
                                                         headbody['b_y{}'.format(p)], headbody['b_x{}'.format(p)], \
                                                         headbody['b_h{}'.format(p)], headbody['b_w{}'.format(p)]
                h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
                b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]
                h_crop = transform(Image.fromarray(h_crop))
                b_crop = transform(Image.fromarray(b_crop))

                # load head and body masks
                mask = torch.zeros([2, 256, 256])  # head, body, gaze location
                mask[0, :, :][int(h_y * 256):int((h_y + h_h) * 256), int(h_x * 256):int((h_x + h_w) * 256)] = 1
                mask[1, :, :][int(b_y * 256):int((b_y + b_h) * 256), int(b_x * 256):int((b_x + b_w) * 256)] = 1

                h_crops[i, ::], b_crops[i, ::], masks[i, ::] = h_crop, b_crop, mask

            h_crops, b_crops, masks = h_crops.to(device), b_crops.to(device), masks.to(device)
            out_map = model(images, h_crops, b_crops, masks)  # model prediction of gaze map
            gt_map = gaussian_smooth(gaze_maps, 21, 5)
            gt_map_sums = gt_map.view(test_b_size, 1, -1).sum(dim=2).unsqueeze(1)  # normalize sum up to 1
            gt_map = (gt_map.view(test_b_size, 1, -1) / gt_map_sums).view(test_b_size, 1, 64, 64)

            try:
                for i in range(5):
                    visualize_result(images.cpu().detach().numpy(), gt_map.cpu().detach().numpy(),
                                     out_map.cpu().detach().numpy(), idx=i)
                    plt.savefig('finetuning_outputs/test_epoch{}_plot{}.jpg'.format(finetuning_e, i + 1))
                    plt.close('all')
            except:
                continue


def evaluate_model_gaze(anno_path, test_img_path, test_bbx_path, model, fig_path):
    '''
    @param anno_path: gazed location
    @param test_img_path: test image path
    @param test_bbx_path: test bounding box path
    @param model: transformer model
    @param fig_path: figure destination path
    @return:
        output: excel sheet with euclidean error and angular error
    '''
    model.eval()

    test_data = GazeDataloader_gazevideo(anno_path, test_img_path, test_bbx_path)
    test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True, num_workers=0)
    test_dataiter = iter(test_dataloader)

    IMAGES = []
    GAZE_START = []
    PREDICT_GAZE = []
    GT_GAZE = []
    for images_name, images, gaze_maps in test_dataiter:
        # images_name, images, gaze_maps = test_dataiter.next()
        images, gaze_maps = images.to(device), gaze_maps.to(device)
        test_b_size = images.shape[0]
        h_crops, b_crops, masks = torch.zeros(images.shape), torch.zeros(images.shape), torch.zeros(
            [test_b_size, 2, 256, 256])
        person_idx = []
        for i in range(test_b_size):  # for each image, train with all gaze-orienting people present in the image
            # load image
            inputs = plt.imread('{}/{}'.format(test_img_path, images_name[i]))
            h, w, _ = inputs.shape
            # load head and body region images
            bbx_name = images_name[i].split('.jpg')[0] + '.json'
            with open('{}/{}'.format(test_bbx_path, bbx_name)) as file:
                headbody = json.load(file)
            num_people = int(len(headbody) / 8)
            # randomly choose one gaze-orienting people to train
            p = random.choice(np.arange(num_people))
            person_idx.append(p)
            h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w = headbody['h_y{}'.format(p)], headbody['h_x{}'.format(p)], \
                                                     headbody['h_h{}'.format(p)], headbody['h_w{}'.format(p)], \
                                                     headbody['b_y{}'.format(p)], headbody['b_x{}'.format(p)], \
                                                     headbody['b_h{}'.format(p)], headbody['b_w{}'.format(p)]
            h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
            b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]
            h_crop = transform(Image.fromarray(h_crop))
            b_crop = transform(Image.fromarray(b_crop))

            # load head and body masks
            mask = torch.zeros([2, 256, 256])  # head, body, gaze location
            mask[0, :, :][int(h_y * 256):int((h_y + h_h) * 256), int(h_x * 256):int((h_x + h_w) * 256)] = 1
            mask[1, :, :][int(b_y * 256):int((b_y + b_h) * 256), int(b_x * 256):int((b_x + b_w) * 256)] = 1

            h_crops[i, ::], b_crops[i, ::], masks[i, ::] = h_crop, b_crop, mask

        h_crops, b_crops, masks = h_crops.to(device), b_crops.to(device), masks.to(device)
        out_map = model(images, h_crops, b_crops, masks)  # model prediction of gaze map
        gt_map = gaussian_smooth(gaze_maps, 21, 5)
        gt_map_sums = gt_map.view(test_b_size, 1, -1).sum(dim=2).unsqueeze(1)  # normalize sum up to 1
        gt_map = (gt_map.view(test_b_size, 1, -1) / gt_map_sums).view(test_b_size, 1, 64, 64)

        # visualization
        for i in range(test_b_size):
            p = person_idx[i]
            bbx_name = images_name[i].split('.jpg')[0] + '.json'
            with open('{}/{}'.format(test_bbx_path, bbx_name)) as file:
                headbody = json.load(file)
            h_y, h_x, h_h, h_w, b_y, b_x, b_h, b_w = headbody['h_y{}'.format(p)], headbody['h_x{}'.format(p)], \
                                                     headbody['h_h{}'.format(p)], headbody['h_w{}'.format(p)], \
                                                     headbody['b_y{}'.format(p)], headbody['b_x{}'.format(p)], \
                                                     headbody['b_h{}'.format(p)], headbody['b_w{}'.format(p)]
            gaze_start, pred_gaze, gt_gaze = plot_gaze((h_y, h_x, h_h, h_w), (b_y, b_x, b_h, b_w),
                                                       images.cpu().detach().numpy(), gt_map.cpu().detach().numpy(),
                                                       out_map.cpu().detach().numpy(), idx=i)
            plt.savefig('{}/{}_result.jpg'.format(fig_path, images_name[i]))
            plt.close('all')
            IMAGES.append(images_name[i])
            GAZE_START.append(gaze_start)
            PREDICT_GAZE.append(pred_gaze)
            GT_GAZE.append(gt_gaze)

        output = pd.DataFrame({'image': IMAGES,
                               'gaze_start_y': np.array(GAZE_START)[:, 0],
                               'gaze_start_x': np.array(GAZE_START)[:, 1],
                               'gazed_y': np.array(GT_GAZE)[:, 0],
                               'gazed_x': np.array(GT_GAZE)[:, 1],
                               'est_y': np.array(PREDICT_GAZE)[:, 0],
                               'est_x': np.array(PREDICT_GAZE)[:, 1]})
    return output


def evaluate_2model(anno_path, test_img_path, test_bbx_path, head_bbx_path, chong_est, model, fig_path,
                    bbx_noise=False):
    '''
    @param anno_path: gazed location
    @param test_img_path: test image path
    @param test_bbx_path: test bounding box path
    @param head_bbx_path: gaze-orienting people bounding box path
    @param chong_est: chong model estimation excel
    @param model: transformer model
    @param fig_path: figure destination path
    @return:
        output: excel sheet with euclidean error and angular error of both models (gaze transformer and chong et 2020)
    '''

    model.eval()
    test_data = GazeDataloader_gazevideo(anno_path, test_img_path, test_bbx_path)
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
    for images_name, images, gaze_maps, img_anno in test_dataiter:
        print(images_name)
        # images_name, images, gaze_maps, img_anno = test_dataiter.next()
        test_b_size = images.shape[0]
        images, gaze_maps = images.to(device), gaze_maps.to(device)
        h_crops, b_crops, masks = torch.zeros(images.shape), torch.zeros(images.shape), torch.zeros(
            [test_b_size, 1, 224, 224])

        # load chong model estimation for the image
        try:
            chong_image_est = chong_est[chong_est['image'] == images_name[0]]
            if len(chong_image_est) == 0:
                continue
        except: pass
        ref_h, ref_w = 600, 800

        # load image
        inputs = plt.imread('{}/{}'.format(test_img_path, images_name[0]))
        h, w, _ = inputs.shape

        try:
            headbody = bbx[
                '/Users/nicolehan/Documents/Research/gazetransformer/gaze_video_data/transformer_all_img/{}'.format(
                    images_name[0])]
        except:
            continue
        num_people = int(len(headbody) / 2)

        # for each image, loop through all the gaze-orienting individual
        for p in range(num_people):
            PERSON_IDX.append(p + 1)
            try:
                h_y, h_x, h_h, h_w = headbody['head{}'.format(p)]
                b_y, b_x, b_h, b_w = headbody['body{}'.format(p)]
                if bbx_noise:
                    h_x += .01
                    b_x += .01
            except:
                continue
            h_crop = inputs[int(h_y * h):int((h_y + h_h) * h), int(h_x * w):int((h_x + h_w) * w)]
            b_crop = inputs[int(b_y * h):int((b_y + b_h) * h), int(b_x * w):int((b_x + b_w) * w)]
            h_crop = transform(Image.fromarray(h_crop))
            b_crop = transform(Image.fromarray(b_crop))

            # find corresponding gaze-orienting in the chong model estimation
            try:
                ref_h_x1, ref_h_x2, ref_h_x3 = chong_image_est['h_x1'].iloc[0] / ref_w, \
                                               chong_image_est['h_x2'].iloc[0] / ref_w, \
                                               chong_image_est['h_x3'].iloc[0] / ref_w
                ref_heads = [ref_h_x1, ref_h_x2, ref_h_x3]
                ref_heads = [i for i in ref_heads if i > 0]
                # print('{}_person{}'.format(images_name[0], p + 1))
                # print(ref_heads)
                ref_index = np.argmin(abs(np.array(ref_heads) - h_x)) + 1

                chong_estx, chong_esty = chong_image_est['model_estx.{}'.format(ref_index)].iloc[0] / ref_w, \
                                         chong_image_est['model_esty.{}'.format(ref_index)].iloc[0] / ref_h
                chong_model_est = np.array([chong_estx, chong_esty])
                # print('{}_person{} no chong model estimation'.format(images_name[0], p + 1))
                # print(chong_model_est)
            except: pass

            # load head and body masks
            masks[0, :, :][int(h_y * 224):int((h_y + h_h) * 224), int(h_x * 224):int((h_x + h_w) * 224)] = 1
            # mask[1, :, :][int(b_y * 224):int((b_y + b_h) * 224), int(b_x * 224):int((b_x + b_w) * 224)] = 1
            h_crops[0, ::], b_crops[0, ::], masks[0, ::] = h_crop, b_crop, masks
            h_crops, b_crops, masks = h_crops.to(device), b_crops.to(device), masks.to(device)
            flips = torch.zeros(1)

            gaze_pred = model(images, h_crops, masks).squeeze(1).detach().numpy()
            gaze_pos = torch.vstack([img_anno['gaze_x'], img_anno['gaze_y']]).permute(1, 0).to(device)
            out_map = torch.zeros([1,1,224,224])
            out_map[0,0, int((gaze_pred[0][1] - .05) * 224):int((gaze_pred[0][1] + .05) * 224),\
                    int((gaze_pred[0][0] - .05) * 224):int((gaze_pred[0][0] + .05) * 224)] = 1
            out_map = gaussian_smooth(out_map, 21, 10)

            gt_map = torch.zeros([1,1,224,224])
            gt_map[0,0, int((gaze_pos[0][1] - .05) * 224):int((gaze_pos[0][1] + .05) * 224),\
                    int((gaze_pos[0][0] - .05) * 224):int((gaze_pos[0][0] + .05) * 224)] = 1
            gt_map = gaussian_smooth(gt_map, 21, 10)
            # visualization
            if os.path.isdir(fig_path) == False:
                os.mkdir(fig_path)
            h_yxhw, b_yxhw = (h_y, h_x, h_h, h_w), (b_y, b_x, b_h, b_w)
            try:
                gaze_start, pred_gaze, gt_gaze = plot_gaze(h_yxhw, b_yxhw, images.cpu().detach().numpy(),
                                                           gt_map.cpu().detach().numpy(), out_map.cpu().detach().numpy(),
                                                           chong_model_est)
            except:
                gaze_start, pred_gaze, gt_gaze = plot_gaze(h_yxhw, b_yxhw, images.cpu().detach().numpy(),
                                                           gt_map.cpu().detach().numpy(),
                                                           out_map.cpu().detach().numpy())
            plt.savefig('{}/{}_person{}_result.jpg'.format(fig_path, images_name[0], p + 1))
            plt.clf()
            plt.close('all')
            IMAGES.append(images_name[0])
            GAZE_START.append(gaze_start)
            PREDICT_GAZE.append(pred_gaze)
            GT_GAZE.append(gt_gaze)
            try: CHONG_PREDICT_GAZE.append(chong_model_est)
            except: pass

    try:
        output = pd.DataFrame({'image': IMAGES,
                               'gaze_start_y': np.array(GAZE_START)[:, 0],
                               'gaze_start_x': np.array(GAZE_START)[:, 1],
                               'gazed_y': np.array(GT_GAZE)[:, 0],
                               'gazed_x': np.array(GT_GAZE)[:, 1],
                               'transformer_est_y': np.array(PREDICT_GAZE)[:, 0],
                               'transformer_est_x': np.array(PREDICT_GAZE)[:, 1],
                               'chong_est_x': np.array(CHONG_PREDICT_GAZE)[:, 0],
                               'chong_est_y': np.array(CHONG_PREDICT_GAZE)[:, 1],
                               })
    except:
        output = pd.DataFrame({'image': IMAGES,
                               'gaze_start_y': np.array(GAZE_START)[:, 0],
                               'gaze_start_x': np.array(GAZE_START)[:, 1],
                               'gazed_y': np.array(GT_GAZE)[:, 0],
                               'gazed_x': np.array(GT_GAZE)[:, 1],
                               'transformer_est_y': np.array(PREDICT_GAZE)[:, 0],
                               'transformer_est_x': np.array(PREDICT_GAZE)[:, 1],
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
    axs.imshow(img)

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
