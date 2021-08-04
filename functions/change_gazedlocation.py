import glob, json
import pandas as pd

vid_info = pd.read_excel('gaze_video_data/Video_Info.xlsx')
head_path = 'gaze_video_data/boundingbox_head (gazed person)'

for i in range(len(vid_info)):
    r = vid_info.iloc[i]
    movie = r['Video']
    bbx_name = head_path + '/' +movie+'.json'
    with open(bbx_name) as file:
        # print('{}/{}'.format(train_bbx_path, bbx_name))
        gazedhead = json.load(file)
    gazed_x = (gazedhead['x']+.5*gazedhead['w'])/1920*800
    gazed_y = (gazedhead['y']+.5*gazedhead['h'])/1080*600

    vid_info.loc[ (vid_info['Video']==movie), 'gazed_locationx'] = gazed_x
    vid_info.loc[(vid_info['Video'] == movie), 'gazed_locationy'] = gazed_y

vid_info.to_excel('video_info.xlsx')