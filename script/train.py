#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:08:54 2021

@author: nicolehan
"""

import torch, os, sys, time
from torch import nn
from numpy import unravel_index
try:
    from utils import *
except:
    pass

def train(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion,
          e_start, num_e, lbd, outpath, b_size=128):
    LOSS = []
    LOSS_TEST = []
    for e in np.arange(e_start, e_start + num_e):
        start_time = time.time()
        model.train()

        print('Epoch:', e, 'Training')
        train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
        if device != 'cpu':
            train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True)
        else:
            train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True)
        train_dataiter = iter(train_dataloader)

        loss_iter = []
        for images_name, images, flips, h_crops, masks, eye, targetgaze, _ in train_dataiter:
            opt.zero_grad()

            images, h_crops, masks, eye, targetgaze = \
                images.to(device), \
                h_crops.to(device), \
                masks.to(device), \
                eye.to(device),\
                targetgaze.to(device)

            b_size = images.shape[0]
            gaze_pred = model(images, h_crops, masks).squeeze(1)

            loss = criterion(gaze_pred, targetgaze)
            # pre_en, pre_de, pre_bbx = model.vit.transformer.encoder.layers[0].state_dict()['self_attn.in_proj_weight'].clone(),\
            #                             model.decoder.layers[0].state_dict()['self_attn.in_proj_weight'].clone(),\
            #                             model.gaze_bbox.layers[0].state_dict()['weight'].clone()
            #plot_grad_flow(model.named_parameters())
            loss.backward() #.retain_grad()
            opt.step()
            # post_en, post_de, post_bbx = model.vit.transformer.encoder.layers[0].state_dict()['self_attn.in_proj_weight'].clone(),\
            #                             model.decoder.layers[0].state_dict()['self_attn.in_proj_weight'].clone(),\
            #                             model.gaze_bbox.layers[0].state_dict()['weight'].clone()
            loss_iter.append(loss.detach().item())

        print("training loss: {:.10f}".format(np.mean(np.array(loss_iter))))
        LOSS.append(np.mean(np.array(loss_iter)))

        # check with test images
        model.eval()
        test_data = GazeDataloader(ann_path, test_img_path, test_bbx_path)
        test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=True)
        test_dataiter = iter(test_dataloader)
        with torch.no_grad():
            loss_iter = []
            for images_name, images, flips, h_crops, masks, eye, targetgaze, _ in test_dataiter:
                images, h_crops, masks, eye, targetgaze = \
                    images.to(device), \
                    h_crops.to(device), \
                    masks.to(device), \
                    eye.to(device), \
                    targetgaze.to(device)

                test_b_size = images.shape[0]
                gaze_pred = model(images, h_crops, masks).squeeze(1)  # model prediction of gaze map
                test_loss = criterion(gaze_pred, targetgaze)
                loss_iter.append(test_loss.detach().item())

            print("testing loss: {:.10f}".format(np.mean(np.array(loss_iter))))
            LOSS_TEST.append(np.mean(np.array(loss_iter)))

        if e % 2==0:
                PATH = "{}/model_epoch{}.pt".format(outpath,e)
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': LOSS,
                    'test_loss': LOSS_TEST,
                }, PATH)
        end_time = time.time()
        print('training epoch time: {:.2f}'.format(end_time-start_time))
        torch.cuda.empty_cache()


