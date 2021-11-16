import os, math, sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
import torchvision.models as models
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from typing import Dict, Iterable, Callable

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append('/Users/nicolehan/Documents/Research/gazetransformer')
try:
    from utils import *
except:
    pass

class ExtractFeatures(nn.Module):
    """Extracts embeddings of segmented humans' heads and bodies from the input images."""

    def __init__(self):
        super(ExtractFeatures, self).__init__()
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        self.shortcut = nn.Identity()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        )
    def forward(self, inputs):
        """Applies extractfetures module.
        By default we use resnet18 as backbone to extract features of the head and body
        Args:
          inputs: input image.
        Returns:
          output: `input feature`
        """
        self.model.eval()
        with torch.no_grad():
            if len(inputs.shape) == 3:  # if it's just one image
                inputs = inputs.unsqueeze(0)
            residual = self.shortcut(inputs)
            extractor = torch.nn.Sequential(*list(self.model.children())[:-2])
            for param in extractor.parameters():
                param.requires_grad = False
            x = extractor(inputs) #[b_size, 512, 7, 7]
        x = self.conv(x) #[b_size, 128, 14, 14]
        x = x.reshape(-1,x.shape[1],x.shape[2]*x.shape[3]) #[b_size, 128, 7*7]
        return x.permute(0,2,1) #to match the vit feature


class SpatialAttention(nn.Module):
    """Extracts spatial attention from heads and body masks."""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # self.project = nn.Sequential(
        #     nn.Linear(784, 392),
        #     # nn.BatchNorm1d(num_features=196)
        #     nn.ReLU(),
        #     nn.Linear(392, 256),
        #     # nn.BatchNorm1d(num_features=196)
        #     nn.ReLU()
        # )
        self.convs = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=3),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # # self.norm = nn.Identity()
        # self.mpl = nn.Sequential(
        #     nn.Linear(32, 32, bias=True),
        #     nn.Linear(32, 16, bias=True),
        #     nn.Linear(16, 2, bias=True),
        # )

        # self.conv1 = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=5)
    def forward(self, masks, h_features, b_features):
        h_spa_feat = h_features + self.convs(masks[:,0,::].unsqueeze(1)).reshape(-1,128,7*7).permute(0,2,1) #[b_size, 7x7, 128]
        b_spa_feat = b_features + self.convs(masks[:, 1, ::].unsqueeze(1)).reshape(-1, 128, 7*7).permute(0, 2, 1) #[b_size, 7x7, 128]
        x = torch.cat([h_spa_feat,b_spa_feat], -1) #[b_size, 14x14, 128x2]
        #img_vit_feature [b_size, 7x7, 256]
        return x



class GazePredictor(nn.Module):
    """Predict final gaze estimation"""
    def __init__(self, hidden_dim=256, nheads=8,
                 num_encoder_layers=3, num_decoder_layers=2):
        super(GazePredictor, self).__init__()
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.GELU(),
            nn.Dropout(.8),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.GELU(),
            nn.Dropout(.8),
            nn.Linear(in_features=128, out_features=2, bias=True)
        )
        self.pos = nn.Parameter(torch.rand(1, hidden_dim))

    def forward(self, hb_spatial, img_vit_out):
        b_size = img_vit_out.shape[1]
        pos = self.pos.unsqueeze(0).repeat(1, b_size, 1) #[1, b_size, 256]
        pos_emb = self.transformer(hb_spatial+img_vit_out, pos) #[1, b_size, 256]
        x = self.mlp(pos_emb)
        return x.squeeze(0)



class Gaze_Transformer(nn.Module):
    """Main Model"""
    def __init__(self):
        super(Gaze_Transformer, self).__init__()
        self.resnet = ExtractFeatures()
        self.spa_net = SpatialAttention()
        self.gaze_pred = GazePredictor()
        self.vit = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(2)

        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def forward(self, images,h_crops,b_crops,masks):
        # h+b feature
        h_features, b_features = self.resnet(h_crops), self.resnet(b_crops) # head feature, body feature #[b_size, 7x7, 128]
        # h,b mask boundingbox as 0, others are 1
        hb_spatial = self.spa_net(1-masks, h_features, b_features).permute(1, 0, 2) #[7*7, b_size, 256]

        # image vit feature
        # vit_encoder = torch.nn.Sequential(*list(self.vit.children())[2].layers[0])
        # vit_encoder = torch.nn.Sequential(*list(list(vit.children())[2].layers))
        # hb_vit_feature = vit_encoder(hb_spatial)
        # vit_feature_extractor = torch.nn.Sequential(*list(self.vit.children())[:-1])
        # img_vit_feature = vit_feature_extractor(images)  # [b_size, 7x7, 256]

        # image vit feature from DETR
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook
        hook1 = get_activation('self_attn')
        self.vit.transformer.encoder.layers[-1].self_attn.register_forward_hook(hook1)
        # h2=self.vit.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook2)
        # h3=self.vit.class_embed.register_forward_hook(hook3)
        output = self.vit(images)
        img_vit_out = activation['self_attn'][0] # [b_size, 49, 256]

        gaze_pos = self.gaze_pred(hb_spatial, img_vit_out)

        return gaze_pos
