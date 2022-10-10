#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:29:32 2022

@author: nicolehan
"""



import os, glob, cv2, sys, torch,natsort, json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns

mycolors = sns.color_palette("colorblind", 10)

vitpath = '/Users/nicolehan/Documents/Research/vIT_gazetransformer'
sys.path.insert(0, vitpath)
sys.path.insert(0, '{}/script'.format(vitpath))

from model import Gaze_Transformer

vidmpath = '/Users/nicolehan/Movies/GazeVideos (free eye movements from the beginning)/exp_script'

# load model
model_condition = 'head' #model trained on head+body, or head, or body
epoch = 20
model = Gaze_Transformer()
checkpoint = torch.load('{}/trainedmodels/{}_vit/model_epoch{}.pt'.format(vitpath, model_condition, epoch), map_location='cpu')
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)
model.eval()
    
    
transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            #                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



def evaluate(basepath, mov_name, estimation_results=False, visualize=True, outputfps=15):

    print('start evaluating movie {}'.format(mov_name))
    stimpath = '{}/{}'.format(basepath, mov_name)
    head_bbx = pd.read_csv('{}/{}.txt'.format(basepath,mov_name), header=None, names=['frame','x1','y1','x2','y2'])
    frames = glob.glob('{}/*'.format(stimpath))
    frames = natsort.natsorted(frames)
    estx, esty, frame = [], [] , []

    inputs = plt.imread(frames[0])
    H, W, _ = inputs.shape
    
    if estimation_results:
        est = pd.read_csv('{}/{}_estimation.csv'.format(basepath, mov_name))
        print('found estimated results already.. skip evaluation')
    
    if visualize:
        #output video
        print('start creating estimation movie...')
        videoname = "{}/{}_estimation.mp4".format(basepath, mov_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(videoname, fourcc, outputfps, (W, H))

    counter = 1
    for frm in frames:
        frm_name = os.path.split(frm)[-1]
        inputs = plt.imread(frm)
        inputs = np.array(cv2.imread(frm))
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = head_bbx[head_bbx['frame']==frm_name][['x1','y1','x2','y2']].iloc[0]
#        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = 2800,500,3200,760
        h_x, h_y = (bbx_x1+bbx_x2)/2, (bbx_y1+bbx_y2)/2 # center of gazer's head for visualization
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbx_x1/W, bbx_y1/H, bbx_x2/W, bbx_y2/H
        
        if not estimation_results: # initial estimation
            # load head and body masks + crops
            masks = torch.zeros([224, 224])
            masks[int(bbx_y1 * 224):int(bbx_y2 * 224), int(bbx_x1 * 224):int(bbx_x2 * 224)] = 1
            masks = masks.unsqueeze(0).unsqueeze(0)
            box_crops = inputs[int(bbx_y1 * H):int(bbx_y2 * H), int(bbx_x1 * W):int(bbx_x2 * W)]
            box_crops = transform(Image.fromarray(box_crops)).unsqueeze(0)
        
            images = transform(Image.open(frm)).unsqueeze(0)
            gaze_pred = model(images, box_crops, masks)
            gaze_pred_bbx = np.array(gaze_pred['pred_boxes'].detach())[0]
            
            x, y = gaze_pred_bbx[0]*W, gaze_pred_bbx[1]*H
            estx.append(x)
            esty.append(y)
            frame.append(counter)
        else: # already have estimation results
            x, y = est[est['frame']==counter][['est_x','est_y']].iloc[0]
            
        counter +=1
        if visualize:
            h_x, h_y, x, y = int(h_x), int(h_y), int(x), int(y)
            cv2.arrowedLine(inputs, (h_x, h_y), (x, y), np.array(mycolors[0])*255, 5)
            video.write(inputs)
    
    if not estimation_results:
        df = pd.DataFrame({'frame':frame, 'est_x':estx, 'est_y':esty})
        df.to_csv('{}/{}_estimation.csv'.format(basepath, mov_name), index=None)
    
    if visualize:
        cv2.destroyAllWindows()
        video.release()
        
    print('Done.')


basepath = '/Users/nicolehan/Documents/Github/attention-target-detection/images'
mov_name = 'skateboard'
evaluate(basepath, mov_name, estimation_results=True,visualize=True, outputfps=15)