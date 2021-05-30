# coding=utf-8
import cv2
import os

allims = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/VOC2007/JPEGImages'
out = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/COCO/train_imgs'
train = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/VOC2007/ImageSets/Main/train.txt'
f = open(train)
for line in f:
    im_path = os.path.join(allims, line[:-1] + '.bmp')
    im = cv2.imread(im_path)
    out_path = os.path.join(out, line[:-1] + '.bmp')
    cv2.imwrite(out_path, im)