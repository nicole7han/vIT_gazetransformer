import os, json, sys, torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system')

from models_imageclips import *
from train_model_imageclips import *


# sys.path.append('/Users/nicolehan/Documents/Research/gazetransformer')
# from model_patches_training.models_imageclips import *
# from model_patches_training.train_model_imageclips import *


def main():
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training from previous checkpoint (default:False)')
    parser.add_argument('--e_start', type=int, default=0,
                        help='starting epoch number (default:0)')
    parser.add_argument('--num_e', type=int, default=30,
                        help='number of epoch iterations, (default:30)')
    parser.add_argument('--b_size', type=int, default=30,
                        help='training batch size, (default:30)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, (default:1e-4)')
    parser.add_argument('--lbd', type=float, default=.7,
                        help='map loss weight, (default:.7)')
    args = parser.parse_args()

    ann_path = "/mnt/bhd/nicoleh/gazetransformer/data/annotations"
    train_img_path = "/mnt/bhd/nicoleh/gazetransformer/data/train"
    train_bbx_path = "/mnt/bhd/nicoleh/gazetransformer/data/train_bbox"
    test_img_path = "/mnt/bhd/nicoleh/gazetransformer/data/test"
    test_bbx_path = "/mnt/bhd/nicoleh/gazetransformer/data/test_bbox"

    # train model
    print('loading model')
    model = Gaze_Transformer()

    lr = args.lr * args.b_size / 128
    beta1 = .9
    opt = optim.Adam(model.parameters(), lr=lr, betas=(beta1, .999))
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss(reduction='mean')

    if args.resume:
        checkpoint = torch.load('models/viTmodel_epoch{}.pt'.format(args.e_start), map_location='cpu')
        loaded_dict = checkpoint['model_state_dict']
        prefix = 'module.'
        n_clip = len(prefix)
        adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                        if k.startswith(prefix)}
        model.load_state_dict(adapted_dict)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print('loaded model epoch {}'.format(args.e_start))

    # CUDA use all cuda devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('available {} devices...'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).to(device)
    LOSS = train(device, model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion,
                 args.e_start + 1, args.num_e, args.lbd, b_size=args.b_size)


if __name__ == '__main__':
    main()
