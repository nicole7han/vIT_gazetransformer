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

    def __init__(self, input_dim, hidden_dim=256, output_dim=2, num_layers=3):
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
                 num_decoder_layers=3, activation='relu'):
        super(Gaze_Transformer, self).__init__()
#        self.vit = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50_dc5', pretrained=True)
        self.vit = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)# (FREEZE NOT UPDATING)
        # self.vit.eval()
        # for param in self.vit.parameters():
        #     param.trainable = False
        for param in self.vit.parameters():
            param.requires_grad = False
        self.backbone = self.vit.backbone

        # decoder (NOT UPDATING?!)
        decoder_layer = TransformerDecoderLayer()
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.query_embed = nn.Embedding(1, d_model)
        self.d_model=d_model
        self.nhead=nhead
        self.dropout=dropout

        # gaze output bbox (UPDATING)
        self.gaze_bbox = MLP(d_model, dim_feedforward, 2, 3) # UPDATING

    def forward(self, images, h_crops, masks):
        '''image vit feature from DETR'''
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook
        # hook1 = get_activation('self_attn')
        hook1 = get_activation('out_proj')

        self.vit.transformer.encoder.layers[-1].self_attn.register_forward_hook(hook1)
        output = self.vit(images)
        img_vit_out = activation['out_proj'][0] # (output[0]:activation,output[1]:weights), 49(7x7 patches) x 1 x 256 (hidden dimension)
        output = self.vit(torch.cat([masks,masks,masks],1))
        mask_vit_out = activation['out_proj'][0]
        memory = img_vit_out + mask_vit_out # encoder output

        ''' encoder output + query embedding -> decoder '''
        _, bs, _ = img_vit_out.shape # 49 x bs x 256
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) #  1 x bs x 256
        tgt = torch.zeros_like(query_embed) # num_queries x b_s x hidden_dim, torch.Size([1, bs, 256])
        vit_mask = torch.zeros([bs,7,7], dtype=torch.bool) # no padding, all False

        # get pos_mebed
        samples = NestedTensor(images, vit_mask).to(device)
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        pos_embed = pos[-1] # bs x 256 x 7 x 7
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # pass to decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed)   # 1 x num_queries x b_s x hidden_dim, torch.Size([1, 1, 5, 256])
        hs = hs.transpose(1,2) # 1 x bs x 1 x 256

        # output gaze bbox
        outputs_coord = self.gaze_bbox(hs).sigmoid()
        return outputs_coord[-1]

