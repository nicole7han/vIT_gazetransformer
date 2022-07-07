import os, math, sys
import torch, copy
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
import torchvision.models as models
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from typing import Optional, List, Dict, Iterable, Callable
from attention_target.model import ModelSpatial
# from utils import NestedTensor


os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append('/Users/nicolehan/Documents/Research/gazetransformer')
try:
    from utils import *
except:
    from script.utils import *
    pass

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        ''' query self attention '''
        q = k = self.with_pos_embed(tgt, query_pos)  # tgt is empty at the beginning # q shape: torch.Size([91, 5, 256])
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,  # tgt2: torch.Size([91, 5, 256])
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        ''' encoder-decoder attention '''
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  # query is object query embedding
                                   key=self.with_pos_embed(memory, pos),  # key is image memoery with position embedding
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        ''' linaer projection '''
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim=256, output_dim=4, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Gaze_Transformer(nn.Module): #only get encoder attention -> a couple layer of transformer -> predict gaze
    """Main Model"""
    def __init__(self, d_model=256, dim_feedforward=2048, nhead=8, dropout=0.1,
                 num_decoder_layers=3, num_classes=1):
        super(Gaze_Transformer, self).__init__()

        # pretrained CHONG attention model
        self.targetatten = ModelSpatial()
        model_dict = self.targetatten.state_dict()
        pretrained_dict = torch.load('attention_target/model_gazefollow.pt', map_location=torch.device('cpu'))
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        self.targetatten.load_state_dict(model_dict)
        for param in self.targetatten.parameters():  # freeze all parameters
            param.requires_grad = False
        self.conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # pretrained DETR
        self.vit = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        for param in self.vit.backbone.parameters(): # freeze resnet backbone
            param.requires_grad = False
        # for param in self.vit.transformer.encoder.layers[:3].parameters(): # freeze first 3 layers of encoders
        #     param.requires_grad = False
        self.backbone = self.vit.backbone


        # encoder (UPDATING)
        modules = list(self.vit.transformer.encoder.layers) # Finetune encdoer
        self.encoder = nn.Sequential(*modules)

        # decoder (UPDATING)
        self.decoder = self.vit.transformer.decoder# Finetune decoder
        self.query_embed = self.vit.query_embed # Finetune query embed
        self.d_model=d_model
        self.nhead=nhead
        self.dropout=dropout

        # gaze output bbox (UPDATING)
        self.class_embed = nn.Linear(d_model, num_classes+1)
        self.gaze_bbox = nn.Sequential(nn.Linear(d_model, d_model, bias=True),
                                       nn.Linear(d_model, d_model, bias=True),
                                       nn.Linear(d_model, 2, bias=True))

    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output
    #     return hook

    def forward(self, images, gazer, masks):
        ''' cropped gazer feature '''
        gazer = self.targetatten.conv1_face(gazer)
        gazer = self.targetatten.bn1_face(gazer)
        gazer = self.targetatten.relu(gazer)
        gazer = self.targetatten.maxpool(gazer)
        gazer = self.targetatten.layer1_face(gazer)
        gazer = self.targetatten.layer2_face(gazer)
        gazer = self.targetatten.layer3_face(gazer)
        gazer = self.targetatten.layer4_face(gazer)
        gazer_feat = self.targetatten.layer5_face(gazer) #torch.Size([bs, 1024, 7, 7])

        ''' gazer feature weighted with masks '''
        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        masks_reduced = self.targetatten.maxpool(self.targetatten.maxpool(self.targetatten.maxpool(masks))).view(-1, 784) #torch.Size([bs, 784])
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        gazer_feat_reduced = self.targetatten.avgpool(gazer_feat).view(-1, 1024) #torch.Size([bs, 1024])
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.targetatten.attn(torch.cat((masks_reduced, gazer_feat_reduced), 1)) # torch.Size([bs, 49])
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7) # torch.Size([bs, 1, 7, 7])

        ''' scene feature weighted with masks '''
        im = torch.cat((images, masks), dim=1)
        im = self.targetatten.conv1_scene(im)
        im = self.targetatten.bn1_scene(im)
        im = self.relu(im)
        im = self.targetatten.maxpool(im)
        im = self.targetatten.layer1_scene(im)
        im = self.targetatten.layer2_scene(im)
        im = self.targetatten.layer3_scene(im)
        im = self.targetatten.layer4_scene(im)
        scene_feat = self.targetatten.layer5_scene(im) #torch.Size([bs, 1024, 7, 7])
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights,
                                            scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

        ''' weighted scene feature + '''
        scene_gazer_feat = torch.cat((attn_applied_scene_feat, gazer_feat), 1) #torch.Size([bs, 2048, 7, 7])

        encoding = self.targetatten.compress_conv1(scene_gazer_feat) # conv from 2048 -> 256 to feed to vit encdoer
        encoding = self.targetatten.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.conv(encoding)
        encoding = self.bn(encoding)
        encoding = self.relu(encoding) # torch.Size([bs, 256, 7, 7])
        memory = encoding.flatten(2).permute(2,0,1) # torch.Size([49, bs, 256])



        '''image vit feature from DETR'''
        # activation = {}
        # # hook1 = get_activation('self_attn')
        # hook1 = self.get_activation('out_proj')
        #
        # self.vit.transformer.encoder.layers[-4].self_attn.register_forward_hook(hook1) #get feature from the 3rd encoder layer
        # output = self.vit(images)
        # img_vit_out = activation['out_proj'][0] # (output[0]:activation,output[1]:weights), 49(7x7 patches) x 1 x 256 (hidden dimension)
        # output = self.vit(torch.cat([masks,masks,masks],1))
        # mask_vit_out = activation['out_proj'][0] # 49 x bs x 256
        #
        # img_vit_out = self.encoder(img_vit_out) # pass it through the last 3 encoders
        # mask_vit_out = self.encoder(mask_vit_out)  # pass it through the last 3 encoders
        #
        # memory = img_vit_out + mask_vit_out # final encoder output

        ''' encoder output + query embedding -> decoder '''
        _, bs, _ = memory.shape # 49 x bs x 256
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) #  1 x bs x 256
        tgt = torch.zeros_like(query_embed).to(device) # num_queries x b_s x hidden_dim, torch.Size([1, bs, 256])
        vit_mask = torch.zeros([bs,7,7], dtype=torch.bool) # no padding, all False

        # get pos_mebed
        samples = NestedTensor(images, vit_mask).to(device)
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        _, pos = self.backbone(samples)
        pos_embed = pos[-1] # bs x 256 x 7 x 7
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # pass to decoder
        mask = torch.zeros([bs,7*7],dtype=torch.bool).to(device)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)   # 1 x num_queries x b_s x hidden_dim, torch.Size([#decoders, 100, bs, 256])
        hs = hs.transpose(1,2) # [#decoders x bs x 100 x 256]

        # output gaze bbox
        outputs_class = self.class_embed(hs)
        outputs_coord = self.gaze_bbox(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out



