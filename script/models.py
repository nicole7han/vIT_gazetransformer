import os, math, sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
import torchvision.models as models
from vit_pytorch import ViT
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from typing import Dict, Iterable, Callable

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sys.path.append('/mnt/bhd/nicoleh/gazetransformer/')
# from utils_imageclips import *
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
            extractor = torch.nn.Sequential(*list(self.model.children())[:-3])
            for param in extractor.parameters():
                param.requires_grad = False
            x = extractor(inputs)
        x = self.conv(x) #[b_size, 128, 14, 14]
        x = x.reshape(-1,x.shape[1],x.shape[2]*x.shape[3]) #[b_size, 256, 14*14]
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
                nn.Upsample(scale_factor=2, mode='bilinear'),
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
        h_spa_feat = h_features + self.convs(masks[:,0,::].unsqueeze(1)).reshape(-1,128,14*14).permute(0,2,1) #[b_size, 14x14, 128]
        b_spa_feat = b_features + self.convs(masks[:, 1, ::].unsqueeze(1)).reshape(-1, 128, 14 * 14).permute(0, 2, 1) #[b_size, 14x14, 128]
        x = torch.cat([h_spa_feat,b_spa_feat], -1) #[b_size, 14x14, 128x2]
        #img_vit_feature [b_size, 14x14, 256]
        return x



class GazePredictor(nn.Module):
    """Predict final gaze estimation"""
    def __init__(self):
        super(GazePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )

        self.softmax = nn.Softmax(dim=1)
    def forward(self, feature_attn): #[b_size, 196, 512]
        x = self.softmax(self.mlp(feature_attn))
        return x



class Gaze_Transformer(nn.Module):
    """Main Model"""
    def __init__(self):
        super(Gaze_Transformer, self).__init__()
        self.resnet = ExtractFeatures()
        self.spa_net = SpatialAttention()
        self.vit = ViT(
                    image_size=224,
                    patch_size=16,
                    num_classes=2,
                    dim=256,
                    depth=6,
                    heads=8,
                    mlp_dim=392,
                    dropout=0.1,
                    emb_dropout=0.1
                )
        self.gaze_pred = GazePredictor()
        # self.vit = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
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

        # # get output from last decoder layer
        # dec_attn_weights = []
        # hooks = [self.vit.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        #         lambda self, input, output: dec_attn_weights.append(output[1])),
        # ]
        # outputs = self.vit(images)  # propogate
        # for hook in hooks:
        #     hook.remove()
        # dec_attn_weights = dec_attn_weights[0]

        # h+b feature
        h_features, b_features = self.resnet(h_crops), self.resnet(b_crops) # head feature, body feature #[b_size, 14x14, 256]
        # hb_features = torch.cat([h_features, b_features],2) #[b_size, 14x14, 256x2]

        # h,b mask boundingbox as 0, others are 1
        # m_features = self.maxpool(self.maxpool(self.maxpool(self.maxpool(1-masks)))).flatten(2) # head body position feature [b_size, 2, 14x14]
        # hb_features = torch.cat([h_features, b_features],2)
        hb_spatial = self.spa_net(1-masks, h_features, b_features) #[b_size, 196, 256]

        # image vit feature
        vit_encoder = torch.nn.Sequential(*list(self.vit.children())[2].layers[0])
        hb_vit_feature = vit_encoder(hb_spatial)

        vit_feature_extractor = torch.nn.Sequential(*list(self.vit.children())[:-1])
        img_vit_feature = vit_feature_extractor(images) # [b_size, 14 x 14, 256]

        # visual feature x spatial attention
        feature_attn = img_vit_feature + hb_vit_feature
        gaze_map = self.gaze_pred(feature_attn)

        return gaze_pos
