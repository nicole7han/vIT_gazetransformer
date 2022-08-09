import os, glob, natsort
import pandas as pd
from PIL import Image
from script.model import *
from statsmodels.formula.api import ols
# from evaluate_models.utils_fine_tuning import *
from functions.data_ana_vis import *
from script.matcher import *
from scipy.ndimage import gaussian_filter
transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

basepath = os.getcwd()
vid_name = 'skateboard'
fig_path = 'demo/{}_outputs'.format(vid_name)
os.makedirs(fig_path, exist_ok=True)

model = Gaze_Transformer()
epoch = 300
checkpoint = torch.load('trainedmodels/model_chong_detr/model_epoch{}.pt'.format(epoch), map_location='cpu')
loaded_dict = checkpoint['model_state_dict']
prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                if k.startswith(prefix)}
model.load_state_dict(adapted_dict)
model.eval()

mode = 'heatmap'
frames = glob.glob('demo/{}/*'.format(vid_name))
frames = natsort.natsorted(frames) #sort by frame number correctly
gazer_box = pd.read_csv('demo/{}.txt'.format(vid_name))
for f_idx, f in enumerate(frames):
    image = np.array(Image.open(f))
    h, w, _ = image.shape
    inputs = transform(Image.open(f)).unsqueeze(0)

    # crop_box top left position (x,y) + width and height
    bbx_x, bbx_y, bbx_w, bbx_h = [2694/w, 507/h, (3240-2694)/w, (1452-507)/h]
    bbx_crop = image[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)]
    bbx_crop = transform(Image.fromarray(bbx_crop)).unsqueeze(0)

    # mask
    mask = np.zeros([h, w])  # head, body, gaze location
    mask[int(bbx_y * h):int((bbx_y + bbx_h) * h), int(bbx_x * w):int((bbx_x + bbx_w) * w)] = 1
    mask = Image.fromarray(mask)
    mask = torch.tensor(np.array(mask.resize([224, 224]))).unsqueeze(0).unsqueeze(0)

    gaze_pred = model(inputs, bbx_crop, mask)
    gaze_pred_logits = np.array(gaze_pred['pred_logits'].detach())  # 1 x 100 x 2
    gaze_pred_bbx = np.array(gaze_pred['pred_boxes'].detach())  # 1 x 100 x 4
    gaze_s_x, gaze_s_y = int((bbx_x + .5 * bbx_w) * w), int((bbx_y + .1 * bbx_h) * h)

    if mode == 'arrow':
        # gaze_logits_dff = gaze_pred_logits[0,:,1]-gaze_pred_logits[0,:,0]
        # final_pred_idx = gaze_logits_dff.argmax() # get maximum logit prediction for gazed location
        final_pred_idx = gaze_pred_logits.argmax(1)[0][0]  # get maximum logit prediction for gazed location
        gaze_bbx = gaze_pred_bbx[0][final_pred_idx]

        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        gaze_pred_x, gaze_pred_y = int(gaze_bbx[0] * w), int(gaze_bbx[1]* h)
        # prediction (blue)
        cv2.arrowedLine(image, (gaze_s_x, gaze_s_y), (gaze_pred_x, gaze_pred_y), (0, 0, 255), 2)
        plt.imshow(image)
        plt.savefig('{}/frame_{}.jpg'.format(fig_path, f_idx))
        plt.close('all')
    elif mode == 'heatmap':
        heatmap = np.zeros([h,w])
        for i in range(len(gaze_pred_logits[0][:,0])):
            logit = gaze_pred_logits[0][:,0][i]
            p = np.exp(logit)/(1+np.exp(logit))
            locx, locy = gaze_pred_bbx[0][i]
            locx, locy = int(locx*w), int(locy*h)
            heatmap[locy-int(.02*w):locy+int(.02*w), locx-int(.02*h):locx+int(.02*h)] = p
        heatmap = heatmap/(heatmap.max())
        heatmap = gaussian_filter(heatmap,40)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(heatmap, alpha=.4)
        ax.set_axis_off()
        plt.savefig('{}/frame_{}.jpg'.format(fig_path, f_idx))
        plt.close('all')