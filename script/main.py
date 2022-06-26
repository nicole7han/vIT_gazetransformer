import os, json, sys, torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy('file_system')
from model import *
from train import *


sys.path.append('/home/xhan01/.cache/torch/hub/facebookresearch_detr_main')
# from model_patches_training.models_imageclips import *
# from model_patches_training.train_model_imageclips import *


def main():
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training from previous checkpoint (default:False)')
    parser.add_argument('--e_start', type=int, default=0,
                        help='starting epoch number (default:0)')
    parser.add_argument('--num_e', type=int, default=50,
                        help='number of epoch iterations, (default:64)')
    parser.add_argument('--b_size', type=int, default=64,
                        help='training batch size, (default:512)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, (default:1e-4)')
    parser.add_argument('--lbd', type=float, default=.7,
                        help='map loss weight, (default:.7)')
    parser.add_argument('--outpath', type=str, default='script_headbody/trainedmodels',
                        help='output path')
    parser.add_argument('--train_datapath', type=str, default='train',
                        help='train path')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    basepath = os.path.abspath(os.curdir)
    ann_path = "{}/data/annotations".format(basepath)
    train_img_path = "{}/data/{}".format(basepath,args.train_datapath)
    train_bbx_path = "{}/data/train_bbox".format(basepath)
    test_img_path = "{}/data/test".format(basepath)
    test_bbx_path = "{}/data/test_bbox".format(basepath)

    # train model
    print('loading model')
    model = Gaze_Transformer()

    lr = args.lr * args.b_size / 64
    print('learning rate = {}'.format(lr))
    beta1 = .9
    opt = optim.Adam(model.parameters(), lr=lr, betas=(beta1, .999), weight_decay=0.0001)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss(reduction='mean')

    if args.resume:
        checkpoint = torch.load('{}/model_epoch{}.pt'.format(args.outpath, args.e_start), map_location='cpu')
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
                 args.e_start + 1, args.num_e, args.lbd, outpath=args.outpath, b_size=args.b_size)


if __name__ == '__main__':
    main()
