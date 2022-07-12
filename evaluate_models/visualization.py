import pandas as pd
from script.model import *
from statsmodels.formula.api import ols
from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *


basepath = '/Users/nicolehan/Documents/Research/gazetransformer'
model = Gaze_Transformer()
epoch= 70
checkpoint = torch.load('trainedmodels/model_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
plt.plot(checkpoint['train_loss'])
plt.plot(checkpoint['test_loss'])
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

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





fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
colors = COLORS * 100
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
   ax = ax_i[0]
   ax.imshow(dec_attn_weights[0, idx].view(h, w))
   ax.axis('off')
   ax.set_title(f'query id: {idx.item()}')
   ax = ax_i[1]
   ax.imshow(im)
   ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              fill=False, color='blue', linewidth=3))
   ax.axis('off')
   ax.set_title(CLASSES[probas[idx].argmax()])
fig.tight_layout()

''' visualize encoder attention map'''
# downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
fact = 16
eyex, eyey = eye[0], eye[1]
idxs = [(int(eyey.item() * 224), int(eyex.item() * 224))]  # the gazer's head position
# create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# add one plot per reference point
gs = fig.add_gridspec(1, 3)
axs = [
    fig.add_subplot(gs[0, 0]),
]
# for each one of the reference points, let's plot the self-attention
for idx_o, ax in zip(idxs, axs):
    idx = (idx_o[0] // fact, idx_o[1] // fact)
    ax.imshow(sattn[..., idx[0], idx[1]].detach().numpy(), cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(images[img_idx, 0, ::].detach().numpy())
for (y, x) in idxs:
    scale = images[0, 0, ::].shape[-1] / images[0, 0, ::].shape[-2]
    x = ((x // fact + 0.5)) * fact
    y = ((y // fact + 0.5)) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r', alpha=0.5))
    fcenter_ax.axis('off')
plt.savefig('{}/{}'.format(attention_path, os.path.split(images_name[0])[-1]))
