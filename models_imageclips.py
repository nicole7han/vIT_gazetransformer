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

# sys.path.append('/mnt/bhd/nicoleh/gazetransformer/')
# from utils_imageclips import *
sys.path.append('/Users/nicolehan/Documents/Research/gazetransformer')
from model_patches_training.utils_imageclips import *


class ExtractFeatures(nn.Module):
    """Extracts embeddings of segmented humans' heads and bodies from the input images."""
    
    def __init__(self):
        super(ExtractFeatures, self).__init__()
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
#        self.layer = model._modules.get('avgpool')
               
        
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
            extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
            inputs_o = extractor(inputs)
        return inputs_o



class SpatialAttention(nn.Module):
    """Extracts spatial attention from heads and body masks."""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.d1 = nn.Linear(64*61*61, 512)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2)) # b_size x 2 x 256 x 256 -> b_size x 32 x 126 x 126
        x = self.bnorm1(x)
        x = F.relu(F.max_pool2d(self.conv2(x),2)) # b_size x 32 x 126 x 126 -> b_size x 64 x 61 x 61
        x = self.bnorm2(x)
        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = F.relu(x)
        return x



class GazePredictor(nn.Module):
    """Predict final gaze estimation"""
    def __init__(self):
        super(GazePredictor, self).__init__()
        self.vit_encoder = nn.Sequential(
            nn.Linear(768, 512, bias=True),
            nn.GELU(),
            nn.Dropout(.5),
        )
        self.mpl1 = nn.Sequential( #shared weights for each patch, across channels
            nn.Linear(512, 64, bias=True),
            nn.GELU(),
            nn.Dropout(.5),
            # nn.Linear(128, 64, bias=True),
            # nn.GELU(),
            # nn.Dropout(.5),
        )
        self.mpl2 = nn.Sequential(#shared weights for each channel, across patches
            nn.Linear(198, 64, bias=True),
            nn.GELU(),
            nn.Dropout(.5)
        )
        self.fc = nn.Linear(64*64, 4096, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, vis_feature, spa_attention, vit_feature):
        spa_atten = spa_attention.unsqueeze(-1)
        x = self.vit_encoder(vit_feature) #b_sizex196x512   # resize vit feature
        x = torch.cat((vis_feature.permute(0,2,1), x),1) #b_sizex198x512   # concatenate image feature and vit feature
        x = (x * spa_atten.permute(0,2,1)) #b_sizex198x512    # multiply to mask feature
        x = self.mpl1(x) #b_sizex198x64  # reduce dimension of each patch
        x = x.permute(0, 2, 1)  #b_sizex64x198
        x = torch.flatten(self.mpl2(x),1) #b_sizex64x64
        x = self.fc(x)
        x = self.softmax(x).view(x.shape[0], 1, 64, 64)

        return x




class Gaze_Transformer(nn.Module):
    """Main Model"""
    def __init__(self):
        super(Gaze_Transformer, self).__init__()
        self.extractor = ExtractFeatures()
        self.spa_net = SpatialAttention()
        self.gaze_pred = GazePredictor()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.softmax = nn.Softmax(dim=1)

        ''' vision transformer
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        ast_hidden_states = outputs.last_hidden_state
        '''

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
    def forward(self, images_name_asc,flips,h_crops,b_crops,masks):
        self.vit.eval()

        #









        config = resolve_data_config({}, model=self.vit)
        transform = create_transform(**config)

        # resnet visual feature
        h_o, b_o = self.extractor(h_crops), self.extractor(b_crops)
        vis_feature = torch.cat((h_o, b_o), 2) # concatenated visual features
        vis_feature = vis_feature.squeeze(-1)
        vit_feature_extractor = torch.nn.Sequential(*list(self.vit.children())[:-1])
        # b_size x 512 x 2
        
        # spatial feature
        spa_attention = self.spa_net(masks) # b_size x 512
        b_size = images_name_asc.shape[0]
        vit_feature = torch.zeros((b_size, 196, 768))
        for i in range(b_size): # get PIL images to get visiontransformer input
            name = ASCII2str(images_name_asc[i])
            flip = flips[i]
            try:
                img = Image.open(name)
            except:
                img = Image.open("gaze_video_data/transformer_all_img/{}".format(name))

            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            inputs = transform(img).unsqueeze(0)
            img_vit_feature = vit_feature_extractor(inputs.to(device)) #input size 3x384x384
            vit_feature[i,::] = img_vit_feature

        # visual feature x spatial attention 
        gaze_map = self.gaze_pred(vis_feature.to(device), spa_attention.to(device), vit_feature.to(device))
    
        return gaze_map
