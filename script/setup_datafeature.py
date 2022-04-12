import torch
import torch.nn as nn
import os, json, base64, random, sys, torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

resnet = models.resnet18(pretrained=True)
resnet.eval()
with torch.no_grad():
    extractor = torch.nn.Sequential(*list(resnet.children())[:-2])
    for param in extractor.parameters():
        param.requires_grad = False
    x = extractor(inputs)  # [b_size, 512, 7, 7]

basepath = os.path.abspath(os.curdir)
mode = 'train'
ann_path = "{}/data/annotations".format(basepath)
img_path = "{}/data/{}".format(basepath,mode)
bbx_path = "{}/data/{}_bbox".format(basepath,mode)

# get all images
img_paths = []
img_folder_list = os.listdir(img_path)
img_folder_list = [f for f in img_folder_list if not f.startswith('.')]
img_folder_list.sort()
img_folder_nums = len(img_folder_list)
for i in range(img_folder_nums):  # each image folder
    img_list = glob.glob('{}/{}/*'.format(img_path, img_folder_list[i]))
    img_list.sort()
    if len(img_list) > 0:
        img_paths += img_list

features = {}
for name in img_paths:
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
    with open('{}/{}_annotation.json'.format(ann_path, mode)) as file:
        eyegaze = json.load(file)
    img_anno = eyegaze['/'.join(name.split('/')[-3:])]

    # load head and body region images
    seg_bbx = np.load("{}/bbx_{}.npy".format(self.bbx_path, folder_name), allow_pickle=True)
    seg_bbx = seg_bbx[()]

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


    name, img, flip, h_crop, b_crop, g_crop, masks, gaze_map,\
                   torch.tensor([img_anno['eye_x'],img_anno['eye_y']]),\
                   torch.tensor([img_anno['gaze_x'],img_anno['gaze_y']])