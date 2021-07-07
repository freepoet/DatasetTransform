
import os
import random
import cv2
trainval_percent = 1
train_percent = 0.5      # train 50%   val 50%
xmlfilepath = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/VOC2007/Annotations'
# txtsavepath = './Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('./Main/trainval.txt', 'w')
ftest = open('./Main/test.txt', 'w')
ftrain = open('./Main/train.txt', 'w')
fval = open('./Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


allims = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/VOC2007/JPEGImages'
out = '/train1'
train = './Main/train.txt'
f = open(train)
for line in f:
    im_path = os.path.join(allims, line[:-1] + '.bmp')
    im = cv2.imread(im_path)
    out_path = os.path.join(out, line[:-1] + '.bmp')
    cv2.imwrite(out_path, im)

allims = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/VOC2007/JPEGImages'
out = '/train2'
train = './Main/val.txt'
f = open(train)
for line in f:
    im_path = os.path.join(allims, line[:-1] + '.bmp')
    im = cv2.imread(im_path)
    out_path = os.path.join(out, line[:-1] + '.bmp')
    cv2.imwrite(out_path, im)