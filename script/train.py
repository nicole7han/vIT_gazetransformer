#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:08:54 2021

@author: nicolehan
"""

import torch, os, sys, time, gc
from torch import nn
from numpy import unravel_index
try:
    from utils import *
except:
    from script.utils import *
    pass

def train_one_epoch(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path,
                    ann_path, opt, criterion, e_start, num_e, e, lbd, outpath, b_size):
    start_time = time.time()
    model.train()

    print('Epoch:', e, 'Training')
    train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
    if device != 'cpu':
        train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True)
    train_dataiter = iter(train_dataloader)

    train_loss_iter = []
    for images_name, images, flips, h_crops, masks, eye, targetgaze, _ in train_dataiter:
        opt.zero_grad()

        images, h_crops, masks, eye, = \
            images.to(device), \
            h_crops.to(device), \
            masks.to(device), \
            eye.to(device)
        b_size = images.shape[0]
        gaze_pred = model(images, h_crops, masks)

        # loss
        loss = criterion(gaze_pred['pred_boxes'], targetgaze['boxes'])
        loss.backward()
        opt.step()
        train_loss_iter.append(loss.detach().item())

    print("training loss: {:.10f}".format(np.mean(np.array(train_loss_iter))))

    # check with test images
    model.eval()
    test_data = GazeDataloader(ann_path, test_img_path, test_bbx_path)
    test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=True)
    test_dataiter = iter(test_dataloader)
    with torch.no_grad():
        test_loss_iter = []
        for images_name, images, flips, h_crops, masks, eye, targetgaze, _ in test_dataiter:
            images, h_crops, masks, eye= \
                images.to(device), \
                h_crops.to(device), \
                masks.to(device), \
                eye.to(device)
            test_b_size = images.shape[0]
            gaze_pred = model(images, h_crops, masks)

            # loss
            test_loss = criterion(gaze_pred['pred_boxes'], targetgaze['boxes'])
            test_loss_iter.append(test_loss.detach().item())

        print("testing loss: {:.10f}".format(np.mean(np.array(test_loss_iter))))

    end_time = time.time()
    print('training epoch time: {:.2f}'.format(end_time - start_time))

    return {'train loss':np.mean(np.array(train_loss_iter)), 'test loss':np.mean(np.array(test_loss_iter))}




def train(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion,
          e_start, num_e, lbd, outpath, b_size):
    LOSS = []
    LOSS_TEST = []
    for e in np.arange(e_start, e_start + num_e):

        train_stats = train_one_epoch(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path,
                                      ann_path, opt, criterion, e_start, num_e, e, lbd, outpath, b_size)
        LOSS.append(train_stats['train loss'])
        LOSS_TEST.append(train_stats['test loss'])

        if e % 10 == 0:
            PATH = "{}/model_epoch{}.pt".format(outpath, e)
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': LOSS,
                'test_loss': LOSS_TEST,
            }, PATH)
