import pandas as pd
from script.model import *
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
# epoch= 70
# checkpoint = torch.load('trainedmodels/model_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
# plt.plot(checkpoint['train_loss'])
# plt.plot(checkpoint['test_loss'])
# loaded_dict = checkpoint['model_state_dict']
# prefix = 'module.'
# n_clip = len(prefix)
# adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
#                 if k.startswith(prefix)}
# model.load_state_dict(adapted_dict)

ann_path = "{}/data/annotations".format(basepath)
train_img_path = "{}/data/train_s".format(basepath)
train_bbx_path = "{}/data/train_bbox".format(basepath)
test_img_path = "{}/data/test".format(basepath)
test_bbx_path = "{}/data/test_bbox".format(basepath)
b_size = 10
train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
train_dataloader = DataLoader(train_data, batch_size= b_size, shuffle=True)
train_dataiter = iter(train_dataloader)
for images_name, images, flips, h_crops, masks, eye, targetgaze, randpos in train_dataiter: #get one batch of train data
    break



''' visuliaze decoder attention weights'''
# for predicting each class (gazed or not gazed), which part of the image is focusing on
# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []
hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    # last layer of encoder
    model.encoder[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    # last layer of decoder
    model.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]
# propagate through the model
outputs = model(images, h_crops, masks)
for hook in hooks:
    hook.remove()
conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0] # bs x 49 x 49
dec_attn_weights = dec_attn_weights[0] # bs x 100 x 49

# keep only predictions with 0.6+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.6

'''  Visualize Decoder Attention  '''
# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]

CLASSES = ['gazed','N/A']
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
colors = COLORS * 100
fig, axs = plt.subplots(ncols=int(keep.sum()), nrows=2, figsize=(7, 7), squeeze=False)
for idx, ax_i in zip(keep.nonzero(), axs.T):
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w).detach().numpy())
    ax.axis('off')
    ax.set_title('decoder attention weights')
    ax = ax_i[1]
    ax.imshow(images[0,0,::])
    ax.axis('off')
    # ax.set_title(CLASSES[probas[idx].argmax()])
fig.tight_layout()


''' Visualize Encoder Attention '''
# get the HxW shape of the feature maps of the CNN
f_map = conv_features['0']
shape = f_map.tensors.shape[-2:]
# and reshape the self-attention to a more interpretable shape
sattn = enc_attn_weights[0].reshape(shape + shape).detach().numpy()
print("Reshaped self-attention:", sattn.shape)

# downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
fact = 32

# let's select 4 reference points for visualization
idxs = [(100, 100), (50, 50), (200, 200), (100, 150),]

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
    ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(images[0,0,::])
for (y, x) in idxs:
    scale = 1 #im.height / img.shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')