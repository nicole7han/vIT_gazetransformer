#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:08:54 2021

@author: nicolehan
"""

import torch, os, sys
from numpy import unravel_index
import numpy as np

# sys.path.append('/mnt/bhd/nicoleh/gazetransformer/')
from utils_imageclips import *


def train(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion,
          e_start, num_e, lbd, b_size=128):
    LOSS = []

    test_data = GazeDataloader(ann_path, test_img_path, test_bbx_path)
    test_dataloader = DataLoader(test_data, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataiter = iter(test_dataloader)

    for e in np.arange(e_start, e_start + num_e):
        model.train()

        print('Epoch:', e, 'Training')
        train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
        train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True)
        train_dataiter = iter(train_dataloader)

        loss_iter = []
        for images_name, images, flips, h_crops, b_crops, g_crops, masks, gaze_maps, img_anno in train_dataiter:
            opt.zero_grad()
            images, h_crops, b_crops, g_crops, masks, gaze_maps = \
                images.to(device), \
                h_crops.to(device), \
                b_crops.to(device), \
                g_crops.to(device), \
                masks.to(device), \
                gaze_maps.to(device)

            b_size = images.shape[0]
            out_map = model(images, h_crops, b_crops, masks)  # model prediction of gaze map

            gt_map = gaussian_smooth(gaze_maps.detach(), 21, 5)
            gt_map = gt_map + .05  # smooth
            gt_map_sums = gt_map.view(b_size, 1, -1).sum(dim=2).unsqueeze(1)  # normalize sum up to 1
            gt_map = (gt_map.view(b_size, 1, -1) / gt_map_sums).view(b_size, 1, 64, 64)
            gt_map = gt_map.to(device)

            # loss between probabilty maps
            map_loss = criterion(out_map.float(), gt_map.float())

            # angle loss
            vec1 = (img_anno['gaze_x'] - img_anno['eye_x'], img_anno['gaze_y'] - img_anno['eye_y'])
            gaze_pred = [unravel_index(out_map.cpu()[i, 0, ::].argmax(), out_map.cpu()[i, 0, ::].shape) for i in
                         range(b_size)]
            gaze_pred = np.array(gaze_pred) / out_map[0, 0, ::].shape[0]
            vec2 = (torch.from_numpy(gaze_pred[:, 1]) - img_anno['eye_x'],
                    torch.from_numpy(gaze_pred[:, 0]) - img_anno['eye_y'])
            ang_loss = 0
            for i in range(b_size):
                v1, v2 = [vec1[0][i] * 1000, vec1[1][i] * 1000], [vec2[0][i] * 1000, vec2[1][i] * 1000]
                unit_vector_1 = v1 / np.linalg.norm(v1)
                unit_vector_2 = v2 / np.linalg.norm(v2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product) * 180 / np.pi
                if torch.isnan(torch.tensor(angle)) == False:
                    ang_loss += angle  # angle in degrees
            ang_loss /= b_size
            if torch.isnan(torch.tensor(ang_loss)):
                print("ang_loss nan")
            # loss
            loss = lbd * map_loss + (1 - lbd) * ang_loss * .00001

            loss.backward()
            opt.step()
            loss_iter.append(loss.detach().item())

        print("training loss: {:.10f}".format(np.mean(np.array(loss_iter))))
        LOSS.append(np.mean(np.array(loss_iter)))

        if (e) % 2 == 0:
            if os.path.isdir('outputs') == False:
                os.mkdir('outputs')

            # check with train images
            try:
                for i in range(5):
                    visualize_result(images_name, flips, g_crops, gt_map, out_map, idx=i)
                    plt.savefig('outputs/viTtrain_epoch{}_plot{}.jpg'.format(e, i + 1))
                    plt.close('all')
            except:
                pass

            # check with test images
            model.eval()
            with torch.no_grad():
                images_name, images, flips, h_crops, b_crops, g_crops, masks, gaze_maps, img_anno = test_dataiter.next()
                images, h_crops, b_crops, g_crops, masks, gaze_maps = \
                    images.to(device), \
                    h_crops.to(device), \
                    b_crops.to(device), \
                    g_crops.to(device), \
                    masks.to(device), \
                    gaze_maps.to(device)

                test_b_size = images.shape[0]

                gt_map = gaussian_smooth(gaze_maps.detach(), 21, 5)
                gt_map = gt_map + .05  # smooth
                gt_map_sums = gt_map.view(test_b_size, 1, -1).sum(dim=2).unsqueeze(1)  # normalize sum up to 1
                gt_map = (gt_map.view(test_b_size, 1, -1) / gt_map_sums).view(test_b_size, 1, 64, 64)
                gt_map = gt_map.to(device)
                out_map = model(images, h_crops, b_crops, masks)  # model prediction of gaze map

                map_loss = criterion(out_map.float(), gt_map.float())
                vec1 = (img_anno['gaze_x'] - img_anno['eye_x'], img_anno['gaze_y'] - img_anno['eye_y'])
                gaze_pred = [unravel_index(out_map.cpu()[i, 0, ::].argmax(), out_map.cpu()[i, 0, ::].shape) for i in
                             range(test_b_size)]
                gaze_pred = np.array(gaze_pred) / out_map[0, 0, ::].shape[0]
                vec2 = (torch.from_numpy(gaze_pred[:, 0]) - img_anno['eye_x'],
                        torch.from_numpy(gaze_pred[:, 1]) - img_anno['eye_y'])
                ang_loss = 0
                for i in range(test_b_size):
                    v1, v2 = [vec1[0][i] * 1000, vec1[1][i] * 1000], [vec2[0][i] * 1000, vec2[1][i] * 1000]
                    unit_vector_1 = v1 / np.linalg.norm(v1)
                    unit_vector_2 = v2 / np.linalg.norm(v2)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    ang_loss += (np.arccos(dot_product) * 180 / np.pi)  # angle in degrees
                ang_loss /= test_b_size
                test_loss = lbd * map_loss + (1 - lbd) * ang_loss * .00001

                PATH = "models/viTmodel_epoch{}.pt".format(e)
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'train_loss': LOSS,
                    'test_loss': test_loss,
                }, PATH)

                try:
                    for i in range(5):
                        visualize_result(images_name, flips, g_crops, gt_map, out_map, idx=i)
                        plt.savefig('outputs/viTtest_epoch{}_plot{}.jpg'.format(e, i + 1))
                        plt.close('all')
                except:
                    pass
