import os, math, sys
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T
import torchvision.models as models
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils_imageclips import *


class ExtractFeatures(nn.Module):
    """Extracts embeddings of segmented humans' heads and bodies from the input images."""
    
    def __init__(self):
        super(ExtractFeatures, self).__init__()
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        self.shortcut = nn.Identity()
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, inputs):
        """Applies extractfetures module.
        By default we use resnet18 as backbone to extract features of the head and body
        Args:
          inputs: input image.
        Returns:
          output: `input feature`
        """
#        h,w,_ = inputs.shape
#        inputs = self.Transform(Image.fromarray(inputs))
        with torch.no_grad():
            if len(inputs.shape)==3: #if it's just one image
                inputs = inputs.unsqueeze(0)
            residual = self.shortcut(inputs)
            extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
            x = extractor(inputs)
        return x


class SpatialAttention(nn.Module):
    """Extracts spatial attention from heads and body masks."""

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.project = nn.Conv2d(in_channels=2, out_channels=768, kernel_size=16, stride=16)
        self.norm = nn.Identity()
        self.mpl = nn.Sequential(
            nn.Linear(196, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
        # self.conv1 = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=5)

    def forward(self, x, pos_embed):
        x = self.project(x).flatten(2).transpose(1, 2)
        x += pos_embed[0, 1:, ].unsqueeze(0)  # [b_size, 14x14, 768]
        x = self.mpl(x.transpose(1, 2)).transpose(1, 2)  # [b_size,  1, 768]
        return x



class GazePredictor(nn.Module):
    """Predict final gaze estimation"""
    def __init__(self):
        super(GazePredictor, self).__init__()
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(768, 512, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(512, 64, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.GELU(),
        )
        self.mpl = nn.Sequential(
            nn.Linear(112 * 112, 64 * 64,  bias=True),
            nn.GELU(),
            nn.Dropout(.1),
        )

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(768, 512, bias=True),
        #     nn.GELU(),
        #     nn.Dropout(.1),
        #     nn.Linear(512, 64, bias=True),
        #     nn.GELU(),
        #     nn.Dropout(.1),
        #     nn.Linear(64, 1, bias=True),
        # )
        # self.fc = nn.Linear(4096, 4096, bias=True)
        self.softmax = nn.Softmax(dim=2)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, feature_attn):
        x = feature_attn.permute(0,2,1).view(feature_attn.shape[0], -1, 14, 14) # [b_size, 768, 14, 14]
        x = torch.flatten(self.deconvs(x),1) # [b_size, 14x14]
        x = self.mpl(x).unsqueeze(1) # [b_size, 1, 14x14]

        # x = self.mlp1(feature_attn) # [b_size, 14x14, 1]
        # x = x.squeeze(2)
        # x = self.fc(x) + .05 # add floor
        x = self.softmax(x).view(x.shape[0], 1, 64, 64) #likelihood

        return x


class Gaze_Transformer(nn.Module):
    """Main Model"""
    def __init__(self, img_size=224, patch_size=16, num_patches=14*14, embed_dim=768):
        super(Gaze_Transformer, self).__init__()
        # self.extractor = ExtractFeatures()
        self.spa_net = SpatialAttention()
        self.gaze_pred = GazePredictor()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.softmax = nn.Softmax(dim=1)


        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
    def forward(self, images,h_crops,b_crops,masks):
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        # b_size = images.shape[0]
        # cls_tokens = torch.cat([self.vit.cls_token]*b_size)
        # patches = self.vit.patch_embed(images)
        #
        # transformer_input = torch.cat((cls_tokens, patches), dim=1) + pos_embed
        # attention = self.vit.blocks[0].attn
        # transformer_input_expanded = attention.qkv(transformer_input)[0]
        #
        # # Split qkv into mulitple q, k, and v vectors for multi-head attantion
        # qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (N=197, (qkv), H=12, D/H=64)
        # q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        # k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        # kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
        # attention_matrix = q @ kT


        ### IDEA: use viT to get each patch feature
        pos_embed = self.vit.pos_embed

        # get image vit feature from each 14x14 patch
        vit_feature_extractor = torch.nn.Sequential(*list(self.vit.children())[:-1])
        img_vit_feature = vit_feature_extractor(images) # [b_size, 14 x 14, 768]

        # binary masks feature
        spatial_attn = self.spa_net(masks, pos_embed)  # [b_size, 1, 768]

        # multiply img_vit_feature to binary masks to get spatial related vit feature
        feature_attn = img_vit_feature * spatial_attn # [b_size, 14 x 14, 768]

        # visual feature x spatial attention 
        gaze_map = self.gaze_pred(feature_attn)
    
        return gaze_map
