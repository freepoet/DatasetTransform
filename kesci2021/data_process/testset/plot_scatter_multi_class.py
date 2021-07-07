import os

import cv2
from PIL import Image
import xml.etree.ElementTree as ET

import numpy as np
from mmcv import list_from_file
from matplotlib import pyplot as plt
import seaborn as sns
from dict_output.dict_save_load import *
import re

plt.close('all')
ball_width = []
ball_height= []

cylinder_width = []
cylinder_height = []
square_cage_width = []
square_cage_height = []
cube_width = []
cube_height = []
circle_cage_width = []
circle_cage_height = []
human_body_width = []
human_body_height = []
metal_bucket_width = []
metal_bucket_height = []
tyre_width = []
tyre_height = []
target_width_including_name=load_dict('./dict_output/target_width_including_name')
target_height_including_name=load_dict('./dict_output/target_height_including_name')
target_area_including_name=load_dict('./dict_output/target_area_including_name')

#
# trainimg_width=np.load("./trainimg_width.npy")
# trainimg_height=np.load("./trainimg_height.npy")
# trainimg_area=np.load("./trainimg_area.npy")
scaling_rate=[]
# str='calibration=0.9863265752792358 tensorflow:Final best valid   0 loss=0.20478513836860657 pr=0.39401692152023315 rate=0.提取  '
# # 匹配“calibration=”后面的数字
# pattern = re.compile(r'(?<=calibration=)\d+\.?\d*')
# a=pattern.findall(str)

j=0
for k,v in target_height_including_name.items():
    pattern = re.compile(r'(?<=imageheight=)\d+\.?\d*')
    image_height=pattern.findall(k)[0]
    scaling_rate.append(int(image_height)/640)
    v=v/scaling_rate[j]
    j=j+1
    if 'ball' in k:
        ball_height.append(v)
    elif 'cylinder' in k:
        cylinder_height.append(v)
    elif 'square cage' in k:
        square_cage_height.append(v)
    elif 'cube' in k:
        cube_height.append(v)
    elif 'circle cage' in k:
        circle_cage_height.append(v)
    elif 'human body' in k:
        human_body_height.append(v)
    elif 'metal bucket' in k:
        metal_bucket_height.append(v)
    elif 'tyre' in k:
        tyre_height.append(v)

i=0
for k, v in target_width_including_name.items():
    pattern = re.compile(r'(?<=imagewidth=)\d+\.?\d*')
    image_width=pattern.findall(k)
    v=v/scaling_rate[i]
    i=i+1
    if 'ball' in k:
        ball_width.append(v)
    elif 'cylinder' in k:
        cylinder_width.append(v)
    elif 'square cage' in k:
        square_cage_width.append(v)
    elif 'cube' in k:
        cube_width.append(v)
    elif 'circle cage' in k:
        circle_cage_width.append(v)
    elif 'human body' in k:
        human_body_width.append(v)
    elif 'metal bucket' in k:
        metal_bucket_width.append(v)
    elif 'tyre' in k:
        tyre_width.append(v)

string = ['ball','cylinder','square_cage','cube','circle_cage','human_body','metal_bucket','tyre']
multiclass_width=[ball_width,cylinder_width,square_cage_width,cube_width,circle_cage_width,human_body_width,metal_bucket_width,tyre_width]
multiclass_height=[ball_height,cylinder_height,square_cage_height,cube_height,circle_cage_height,human_body_height,metal_bucket_height,tyre_height]

plt.figure(1)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(axis='both',which='major',labelsize=14)
my_x_ticks = np.arange(0,120, 10)
my_y_ticks = np.arange(0,90, 10)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

font={'family':'serif', 'style':'italic', 'weight':'normal', 'color':'k', 'size':10 } #调用方式如下： plt.plot(x,y,fontdict=font)
for i, str in enumerate(string):
    plt.subplot(241+i)
    x=multiclass_width[i]
    y=multiclass_height[i]
    plt.xlabel(str+'_width', fontdict=font, fontsize=8)
    plt.ylabel(str+'_height', fontdict=font, fontsize=8)
    plt.scatter(x,y,c='k')
plt.show()

plt.figure(2)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(axis='both',which='major',labelsize=14)
my_x_ticks = np.arange(0,8, 1)
my_y_ticks = np.arange(0,1600, 100)
plt.xticks(my_x_ticks, string)
plt.yticks(my_y_ticks)
plt.xlabel('class', fontdict=font, fontsize=15)
plt.ylabel('numbers', fontdict=font, fontsize=15)
number_every_class=[]
for i, str in enumerate(string):
    number_every_class.append(len(multiclass_width[i]))
plt.bar(np.arange(len(number_every_class)),number_every_class)
plt.show()

plt.figure(3)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(axis='both',which='major',labelsize=14)
my_x_ticks = np.arange(0,10, 0.5)
my_y_ticks = np.arange(0,200, 20)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('aspect_ratio', fontdict=font, fontsize=15)
plt.ylabel('numbers', fontdict=font, fontsize=15)
plt.grid(True)

aspect_ratio=[]
for i in np.arange(len(multiclass_width)):
    list2=multiclass_width[i]
    list1=multiclass_height[i]
    list3=np.divide(np.array(list1),np.array(list2))
    aspect_ratio.append(list3)

aspect_ratio_1d=[]
for i in np.arange(len(aspect_ratio)):
    for j in np.arange(len(aspect_ratio[i])):
        aspect_ratio_1d.append(aspect_ratio[i][j])
n, bins, patches = plt.hist(aspect_ratio_1d,500,(0,9),density = False)
plt.show()