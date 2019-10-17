#!/usr/bin/env python
# coding: utf-8

# # Apply cnn model on videos

# In[ ]:


from fastai.vision import *
import seaborn as sns
from matplotlib.pyplot import figure
import torch
import os
from fastai import *
from fastai.vision import *
from IPython.display import HTML
import cv2

# ! pip install pytube
from pytube import YouTube
# ! conda install -c conda-forge imageio
import imageio
# ! pip install moviepy
from moviepy.editor import *

from ipywidgets import Video
from PIL import Image


class Selectbest():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

        
def my_predictor(image_input):
    jif=Image.fromarray(image_input)
    jif.save(vid_output_path+'/temp.jpg')
    image_fai=open_image(vid_output_path+'/temp.jpg')
    prediction,label,prob=learn.predict(image_fai)
    str_prediction=str(prediction)
    selectbest.recent_fit.append(str_prediction)
    if len(selectbest.recent_fit) > 20:
        selectbest.recent_fit = selectbest.recent_fit[1:]
    selectbest.avg_fit=most_frequent(selectbest.recent_fit) 
    im = cv2.imread(vid_output_path+'/temp.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    test.append(selectbest.avg_fit)
    
#     
    cv2.putText(im, selectbest.avg_fit , (50,350), font, 1.1, (255, 10, 10), 2, cv2.LINE_AA)
    
    return im
selectbest = Selectbest()
    
    
def cnn_video_annotator(video_clip_path,vid_output_path,model)   
    video_clip_path = VideoFileClip(video_clip_path)
    predicted_clip= video_clip_path.fl_image(my_predictor)
    vid_output_path = vid_output_path
    predicted_clip.write_videofile(vid_output_path,audio=False)
    return print('video saved at {}'.format(vid_output_path))


# # Run cnn_video_annotator(video_clip_path,vid_output_path,model) and it will return video path 

# In[ ]:




