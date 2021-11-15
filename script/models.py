import os, math, sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
import torchvision.models as models
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# sys.path.append('/mnt/bhd/nicoleh/gazetransformer/')
# from utils_imageclips import *
sys.path.append('/Users/nicolehan/Documents/Research/gazetransformer')
try:
    from utils import *
except:
    pass
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 1))
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

class ExtractFeatures(nn.Module):
    """Extracts embeddings of segmented humans' heads and bodies from the input images."""

    def __init__(self):
        super(ExtractFeatures, self).__init__()
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        self.shortcut = nn.Identity()

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
            if len(inputs.shape) == 3:  # if it's just one image
                inputs = inputs.unsqueeze(0)
            residual = self.shortcut(inputs)
            extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
            x = extractor(inputs).squeeze(-1).permute(0,2,1)
        return x


class SpatialAttention(nn.Module):
    """Extracts spatial attention from heads and body masks."""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(512, 784),
            nn.ReLU()
        )
        # self.project2 = nn.Sequential(
        #     nn.Linear(784, 196),
        #     nn.ReLU()
        # )
        # self.project = nn.Conv2d(in_channels=2, out_channels=768, kernel_size=3, stride=3)
        # self.norm = nn.Identity()
        self.mpl = nn.Sequential(
            nn.Linear(1568, 768),
            nn.ReLU(),
        )
        # self.conv1 = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=5)
    def forward(self, m_features, hb_features):
        x = (self.project(hb_features) + m_features).flatten(1)
        x = self.mpl(x).unsqueeze(1)  # [b_size,  1, 768]
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
        x = feature_attn.permute(0,2,1).view(feature_attn.shape[0], -1, 14, 14) # [b_size, feature_dim 768, 14, 14]
        x = torch.flatten(self.deconvs(x),1) # [b_size, 14x14]
        x = self.mpl(x).unsqueeze(1) # [b_size, 1, 14x14]

        # x = self.mlp1(feature_attn) # [b_size, 14x14, 1]
        # x = x.squeeze(2)
        # x = self.fc(x) + .05 # add floor
        x = self.softmax(x).view(x.shape[0], 1, 64, 64) #likelihood
        return x




class Gaze_Transformer(nn.Module):
    """Main Model"""
    def __init__(self):
        super(Gaze_Transformer, self).__init__()
        self.resnet = ExtractFeatures()
        self.spa_net = SpatialAttention()
        self.gaze_pred = GazePredictor()
        self.vit = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(2)





        ''' vision transformer
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        ast_hidden_states = outputs.last_hidden_state
        '''
        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
                
    def forward(self, images,h_crops,b_crops,masks):
        self.vit.eval()
        self.resnet.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

        # get image vit feature from each 14x14 patch
        # vit_feature_extractor = list(self.vit.children())[0] #just get encoder decoder
        # vit_encoder = torch.nn.Sequential(*list(list(vit_feature_extractor.children())[0].children())[0])

        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            self.vit.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            self.vit.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.vit.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
        outputs = self.vit(images)  # propogate
        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]
        # output of the CNN
        f_map = conv_features['0']
        print("Encoder attention:      ", enc_attn_weights[0].shape)
        print("Feature map:            ", f_map.tensors.shape)
        # get the HxW shape of the feature maps of the CNN
        shape = f_map.tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        img_idx = 2
        sattn = enc_attn_weights[img_idx].reshape(shape + shape)
        print("Reshaped self-attention:", sattn.shape)

        # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
        fact = 32
        idxs = [(10, 10), (50, 50), (100, 100), (150, 150), ]
        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        axs = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[0, -1]),
            fig.add_subplot(gs[1, -1]),
        ]
        # for each one of the reference points, let's plot the self-attention
        # for that point
        for idx_o, ax in zip(idxs, axs):
            idx = (idx_o[0] // fact, idx_o[1] // fact)
            ax.imshow(sattn[..., idx[0], idx[1]].detach().numpy(), cmap='cividis', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'self-attention{idx_o}')

        # and now let's add the central image, with the reference points as red circles
        fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        fcenter_ax.imshow(images[img_idx,0,::].detach().numpy())
        for (y, x) in idxs:
            scale = images[0,0,::].shape[-1]/ images[0,0,::].shape[-2]
            x = ((x // fact) + 0.5) * fact
            y = ((y // fact) + 0.5) * fact
            fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
            fcenter_ax.axis('off')






        # vit_feature_extractor = torch.nn.Sequential(*list(self.vit.children())[:-1])
        # img_vit_feature = vit_feature_extractor(images) # [b_size, 14 x 14, 768]

        # binary masks feature
        h_features, b_features = self.resnet(h_crops), self.resnet(b_crops) # head feature, body feature [b_size, 1, 512]
        hb_features = torch.cat([h_features, b_features],1) #[b_size, 2, 512]
        # mask boundingbox as 0, others are 1
        m_features = self.maxpool(self.maxpool(self.maxpool(1-masks))).flatten(2) # head body position feature [b_size, 2, 784]
        spatial_attn = self.spa_net(m_features, hb_features) #[b_size, 1, 768]

        # # multiply img_vit_feature to binary masks to get spatial related vit feature
        feature_attn = img_vit_feature * spatial_attn  # [b_size, 14 x 14 + 1, 768]

        # visual feature x spatial attention
        gaze_map = self.gaze_pred(feature_attn)

        return gaze_map
