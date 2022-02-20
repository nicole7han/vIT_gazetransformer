import os, json, sys, torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

basepath = os.path.abspath(os.curdir)
from script.model import *
from script.train import *
from script.utils import *

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


ann_path = "{}/data/annotations".format(basepath)
train_img_path = "{}/data/train".format(basepath)
train_bbx_path = "{}/data/train_bbox".format(basepath)
test_img_path = "{}/data/test".format(basepath)
test_bbx_path = "{}/data/test_bbox".format(basepath)
# segmask_path = "/Users/nicolehan/Documents/Research/Gaze Transformer Model with Body Component/CDCL-human-part-segmentation-master/gazefollow/train_person_masks"
#
# cleanup_dataset(segmask_path, bbx_path, img_path)


b_size = 1
train_data = GazeDataloader(ann_path, train_img_path, train_bbx_path)
train_dataloader = DataLoader(train_data, batch_size= b_size, shuffle=True)
train_dataiter = iter(train_dataloader)
#
images_name, images, flips, h_crops, b_crops, g_crops, masks, gaze_maps, img_anno, targetgaze = train_dataiter.next() #get one batch of train data
model = Gaze_Transformer()
model.to(device)
#gaze_pred = model(images, h_crops, b_crops, masks)



## DTER VISUALIZATION
# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []
hooks = [
    model.vit.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.vit.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.vit.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]
outputs = model.vit(images)  # propogate
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
h, w = shape
# and reshape the self-attention to a more interpretable shape
img_idx = 0
im = Image.open(images_name[img_idx])
im = im.resize([244,244])
sattn = enc_attn_weights[img_idx].reshape(shape + shape)
print("Reshaped self-attention:", sattn.shape)

probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9
bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], images[img_idx, 0, ::].detach().numpy().shape)


''' visuliaze decoder attention weights'''
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
fact = 32
eyex, eyey = img_anno['eye_x'], img_anno['eye_y']
idxs = [(int(eyey.item()*244), int(eyex.item()*244))] # the gazer's head position
#idxs = [(10, 10), (50, 50), (100, 100), (150, 150), ]
# here we create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# and we add one plot per reference point
gs = fig.add_gridspec(1, 4)
axs = [
    fig.add_subplot(gs[0, 0]),
#    fig.add_subplot(gs[1, 0]),
#    fig.add_subplot(gs[0, -1]),
#    fig.add_subplot(gs[1, -1]),
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
fcenter_ax.imshow(images[img_idx, 0, ::].detach().numpy())
for (y, x) in idxs:
    scale = images[0, 0, ::].shape[-1] / images[0, 0, ::].shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')




# register the forward hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook
hook1 = get_activation('self_attn')
hook2 = get_activation('multihead_attn')
# hook3 = get_activation('Linear')
h1 = self.vit.transformer.encoder.layers[-1].self_attn.register_forward_hook(hook1)
h2 = self.vit.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook2)
# h3=self.vit.class_embed.register_forward_hook(hook3)
output = self.vit(images)
encoder_out = activation['self_attn'][0].permute(1, 0, 2)  # [b_size, 49, 256]
decoder_out = activation['multihead_attn'][0].permute(1, 0, 2)  # [b_size, 100, 256]



# get output from last decoder layer
dec_attn_weights = []
hooks = [self.vit.transformer.encoder.layers[-1].self_attn.register_forward_hook(
    lambda self, input, output: dec_attn_weights.append(output[1])),
    # self.vit.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
    # lambda self, input, output: dec_attn_weights.append(output[1])),
]
outputs = self.vit(images)  # propogate
for hook in hooks:
    hook.remove()
dec_attn_weights = dec_attn_weights[0]


# VIT check
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
b_size = images.shape[0]
cls_tokens = torch.cat([vit.cls_token] * b_size)
patches = vit.patch_embed(images)
pos_embed = vit.pos_embed
transformer_input = torch.cat((cls_tokens, patches), dim=1) + pos_embed
attention = vit.blocks[0].attn
transformer_input_expanded = attention.qkv(transformer_input)[0]

# Split qkv into mulitple q, k, and v vectors for multi-head attantion
qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (N=197, (qkv), H=12, D/H=64)
q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
attention_matrix = q @ kT

# Visualize attention matrix
img = np.array(Image.open(images_name[0]))
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Attention", fontsize=24)
fig.add_axes()
ax = fig.add_subplot(3, 5, 1)
patch_idx = 55
row, col = round(patch_idx / 14), patch_idx % 14
cv2.rectangle(img, ((col - 1) * 16, row * 16), (col * 16, (row + 1) * 16), (255, 0, 0), 2)
ax.imshow(img)

for i in range(12):  # visualize the 100th rows of attention matrices in the 0-7th heads
    attn_heatmap = attention_matrix[i, patch_idx, 1:].reshape((14, 14)).detach().cpu().numpy()
    ax = fig.add_subplot(3, 5, i + 2)
    ax.imshow(attn_heatmap)

x = transformer_input.clone()
for i, blk in enumerate(vit.blocks):
    # print("Entering the Transformer Encoder {}".format(i))
    x = blk(x)
x = vit.norm(x)
transformer_output = x[:, 0]
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))
print("Classification head: ", vit.head)
result = vit.head(transformer_output)

# Visualize attention matrix
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Visualization of Classification Result", fontsize=24)
fig.add_axes()
for i in range(9):
    result_label_id = int(torch.argmax(result[i]))
    plt.title(imagenet_labels[result_label_id])
    # plt.title("Inference result : id = {}, label name = {}".format(
    #     result_label_id, imagenet_labels[result_label_id]))
    img = Image.open(images_name[i])
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(img)





#train model
model = Gaze_Transformer()
lr = .0001
beta1 = .9
lmbda = .0001
opt = optim.Adam(model.parameters(), lr=lr, betas = (beta1, .999))
# criterion = nn.BCELoss(reduction='mean')
criterion = nn.MSELoss()
e_start = 0

# checkpoint = torch.load('models/model_epoch{}.pt'.format(e_start), map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# opt.load_state_dict(checkpoint['optimizer_state_dict'])
# for state in opt.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.to(device)
num_e = 5
LOSS = train(model, train_img_path, train_bbx_path, test_img_path, test_bbx_path, ann_path, opt, criterion, e_start+1, num_e, b_size=128)



# #evaluate model
# model = Gaze_Transformer()
# lr = .0001
# beta1 = .5
# lmbda = .0001
# opt = optim.Adam(model.parameters(), lr=lr, betas = (beta1, .999))
# criterion = nn.BCELoss()
#
# epoch = 40
# checkpoint = torch.load('models/model_epoch{}.pt'.format(epoch), map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# opt.load_state_dict(checkpoint['optimizer_state_dict'])
# for state in opt.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.to(device)
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.to(device)
#
# evaluate_model(model, test_img_path, test_bbx_path, ann_path, b_size=100)
