import os, json, sys, torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
from models_imageclips import *
from train_model_imageclips import *
from utils_imageclips import *


ann_path = "{}/data/annotations".format(basepath)
train_img_path = "{}/data/train".format(basepath)
train_bbx_path = "{}/data/train_bbox".format(basepath)
test_img_path = "{}/data/test".format(basepath)
test_bbx_path = "{}/data/test_bbox".format(basepath)
# segmask_path = "/Users/nicolehan/Documents/Research/Gaze Transformer Model with Body Component/CDCL-human-part-segmentation-master/gazefollow/train_person_masks"
#
# cleanup_dataset(segmask_path, bbx_path, img_path)


b_size = 10
train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
train_dataloader = DataLoader(train_data, batch_size= b_size, shuffle=True)
train_dataiter = iter(train_dataloader)
#
images, flips, h_crops, b_crops, g_crops, masks, gaze_maps, img_anno = train_dataiter.next() #get one batch of train data
model = Gaze_Transformer()
model.to(device)
images_name_asc = [str2ASCII(name) for name in images_name]
images_name_asc = torch.tensor(images_name_asc).to(device)
b_size = images_name_asc.shape[0]
out_map = model(images_name_asc, flips, h_crops, b_crops, masks)
out_map = out_map.cpu()
# visualize_result(images_name, g_crops, gaze_maps, out_map, idx=0)

#train model
model = Gaze_Transformer()
lr = .0001
beta1 = .9
lmbda = .0001
opt = optim.Adam(model.parameters(), lr=lr, betas = (beta1, .999))
criterion = nn.BCELoss(reduction='mean')
# criterion = nn.MSELoss()
e_start = 0

# checkpoint = torch.load('models/model_epoch{}.pt'.format(e_start), map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# opt.load_state_dict(checkpoint['optimizer_state_dict'])
# for state in opt.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.to(device)
num_e = 5
LOSS = train(model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion, e_start+1, num_e, b_size=128)



# #evaluate model
# model = Gaze_Transformer()
# lr = .0001
# beta1 = .5
# lmbda = .0001
# opt = optim.Adam(model.parameters(), lr=lr, betas = (beta1, .999))
# criterion = nn.BCELoss()
#
# epoch = 40
# checkpoint = torch.load('models/model_epoch{}.pt'.format(epoch), map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# opt.load_state_dict(checkpoint['optimizer_state_dict'])
# for state in opt.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.to(device)
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.to(device)
#
# evaluate_model(model, test_img_path, test_bbx_path, ann_path, b_size=100)
