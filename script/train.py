#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:08:54 2021

@author: nicolehan
"""

import torch, os, sys
from torch import nn
from numpy import unravel_index
try:
    from utils import *
except:
    from script.utils import *


def train(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion,
          e_start, num_e, lbd, b_size=128):
    LOSS = []
    test_data = GazeDataloader(ann_path, test_img_path, test_bbx_path)
    test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=True)
    test_dataiter = iter(test_dataloader)

    for e in np.arange(e_start, e_start + num_e):
        model.train()

        print('Epoch:', e, 'Training')
        train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
        train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True)
        train_dataiter = iter(train_dataloader)

        loss_iter = []
        for images_name, images, flips, h_crops, b_crops, g_crops, masks, gaze_maps, img_anno, targetgaze in train_dataiter:
            opt.zero_grad()
            img_anno['gaze_x']
            images, h_crops, b_crops, g_crops, masks, gaze_maps, targetgaze = \
                images.to(device), \
                h_crops.to(device), \
                b_crops.to(device), \
                g_crops.to(device), \
                masks.to(device), \
                gaze_maps.to(device), \
                targetgaze.to(device)

            b_size = images.shape[0]
            gaze_pred = model(images, h_crops, b_crops, masks)

            '''
            out_map = model(images, h_crops, b_crops, masks)  # model prediction of gaze map
            gaze_maps[gaze_maps>0] = 1
            gaze_maps[gaze_maps==1] = 1-.05 #label smoothing
            gaze_maps[gaze_maps==0] = .05
            gaze_maps = gaussian_smooth(gaze_maps.detach(), 21, 5).to(device)
            gt_map_sums = gaze_maps.view(b_size, 1, -1).sum(dim=2).unsqueeze(1)  # normalize sum up to 1
            gaze_maps = (gaze_maps.view(b_size, 1, -1) / gt_map_sums).view(b_size, 1, 64, 64)
            # loss between probabilty maps
            loss = criterion(out_map.float(), gaze_maps.float())
            '''
            # angle loss
            vec1 = (img_anno['gaze_x'] - img_anno['eye_x'], img_anno['gaze_y'] - img_anno['eye_y'])
            vec2 = (gaze_pred[:, 1] - img_anno['eye_x'],
                    gaze_pred[:, 0] - img_anno['eye_y'])
            ang_loss = 0
            for i in range(b_size):
                v1, v2 = torch.stack([vec1[0][i], vec1[1][i]]), \
                         torch.stack([vec2[0][i], vec2[1][i]])
                unit_vector_1 = v1 / torch.linalg.norm(v1)
                unit_vector_2 = v2 / torch.linalg.norm(v2)
                dot_product = torch.dot(unit_vector_1, unit_vector_2)
                angle = torch.arccos(dot_product) * 180 / np.pi
                if torch.isnan(angle) == False:
                    ang_loss += angle  # angle in degrees
                else:
                    print('{} angle loss nan'.format(images_name[i]))
            ang_loss /= b_size
            loss = lbd * criterion(gaze_pred, targetgaze) + (1 - lbd) * ang_loss * .01

            loss.backward()
            opt.step()
            loss_iter.append(loss.detach().item())

        print("training loss: {:.10f}".format(np.mean(np.array(loss_iter))))
        LOSS.append(np.mean(np.array(loss_iter)))

        if (e) % 2 == 0:
            if os.path.isdir('script3/outputs') == False:
                os.mkdir('script3/outputs')

            # # check with train images
            # try:
            #     for i in range(5):
            #         visualize_result(images_name, flips, g_crops, gaze_maps, gaze_pred, idx=i)
            #         plt.savefig('script3/outputs/ResviTtrain_epoch{}_plot{}.jpg'.format(e, i + 1))
            #         plt.close('all')
            # except:
            #     pass

            # check with test images
            model.eval()
            with torch.no_grad():
                try:
                    images_name, images, flips, h_crops, b_crops, g_crops, masks, gaze_maps, img_anno, targetgaze= test_dataiter.next()
                    images, h_crops, b_crops, g_crops, masks, gaze_maps, targetgaze = \
                        images.to(device), \
                        h_crops.to(device), \
                        b_crops.to(device), \
                        g_crops.to(device), \
                        masks.to(device), \
                        gaze_maps.to(device), \
                        targetgaze.to(device)

                    test_b_size = images.shape[0]
                    gaze_pred = model(images, h_crops, b_crops, masks)  # model prediction of gaze map

                    vec1 = (img_anno['gaze_x'] - img_anno['eye_x'], img_anno['gaze_y'] - img_anno['eye_y'])
                    vec2 = (gaze_pred[:, 1] - img_anno['eye_x'],
                            gaze_pred[:, 0] - img_anno['eye_y'])
                    ang_loss = 0
                    for i in range(b_size):
                        v1, v2 = torch.stack([vec1[0][i], vec1[1][i]]), \
                                 torch.stack([vec2[0][i], vec2[1][i]])
                        unit_vector_1 = v1 / torch.linalg.norm(v1)
                        unit_vector_2 = v2 / torch.linalg.norm(v2)
                        dot_product = torch.dot(unit_vector_1, unit_vector_2)
                        angle = torch.arccos(dot_product) * 180 / np.pi
                        if torch.isnan(angle) == False:
                            ang_loss += angle  # angle in degrees
                        else:
                            print('{} angle loss nan'.format(images_name[i]))
                    ang_loss /= b_size
                    test_loss = lbd * criterion(gaze_pred, targetgaze) + (1 - lbd) * ang_loss * .01
                    print('test_loss : {}'.format(test_loss))

                    PATH = "script3/trainedmodels/resviTmodel_epoch{}.pt".format(e)
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'train_loss': LOSS,
                        'test_loss': test_loss,
                    }, PATH)
                except:
                    PATH = "script3/trainedmodels/resviTmodel_epoch{}.pt".format(e)
                    torch.save({
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'train_loss': LOSS,
                    }, PATH)

                # try:
                #     for i in range(5):
                #         visualize_result(images_name, flips, g_crops, gaze_maps, gaze_pred, idx=i)
                #         plt.savefig('script3/outputs/resviTmodel_epoch{}_plot{}.jpg'.format(e, i + 1))
                #         plt.close('all')
                # except:
                #     pass
