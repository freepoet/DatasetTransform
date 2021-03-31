# -*- coding: utf-8 -*-
"""
@File    : test01.py
@Time    : 2021/3/30 上午10:43
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
2021/3/30 上午10:43      1.0         None
# @Software: PyCharm
"""
import torch
# torch.cuda.set_device(0)
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from test02 import *
import ipdb
#读取MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-255  转为  0-1
    transforms.Normalize((0.5,),(0.5,))  # 0-1  转为  -1-1
])
trainset=torchvision.datasets.MNIST(
    root='E:/HardDisk/data/',
    train=True,
    download=False,
    transform=transform
)
trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
testset=torchvision.datasets.MNIST(
    root='E:/HardDisk/data/',
    train=False,
    download=False,
    transform=transform
)
testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
# dataiter=iter(trainloader)
# images,labels=next(dataiter)
# print(type(labels))
# for j, train_img in enumerate(trainloader, 0):
    # print((train_img[0].size[0]))
    # img_pil=show(torchvision.utils.make_grid((train_img[0]+1)/2)).resize((100,100))
    # # plt.imshow(img_pil)
    # # plt.show()
#     break
acc = []
index_of_max_value = []
correlation = []
right_num = 0
# 尺度变化
start = 0.5
for k in np.arange(start, 2, 0.1):
    k = round(k, 2)  # 保留两位小数
    for i, test_data in enumerate(testloader, 0):
        if i < 10:  # 仅仅测试10张图片
            # 二维list
            correlation.append([])
            # 原始图片为NCWH的tensor 压缩成WH的tensor
            test_img = (test_data[0].squeeze(0)).squeeze(0)
            test_label = test_data[1]
            # 获取测试图片的宽 高
            width = test_img.shape[0]
            height = test_img.shape[1]
            img1 = test_img
            img2 = []
            train_label = []
            for j, train_data in enumerate(trainloader, 0):
                # 原始图片为NCWH的tensor 压缩成WH的tensor
                train_img = (train_data[0].squeeze(0)).squeeze(0)
                train_label.append(train_data[1])
                # 训练图片resize成测试图片一样的大小 将训练图片转换为PIL
                train_img_pil = transforms.ToPILImage()(train_img)
                # 按照拉伸比例 resize图片
                new_width = width
                new_height = round(height*k)
                train_img_resize = train_img_pil.resize((new_width, new_height), Image.ANTIALIAS) #PIL 格式图片才能resize
                # PIL转tensor 3维转2维
                train_img_resize = transforms.ToTensor()(train_img_resize)
                train_img_resize.squeeze_()
                # Tensor转ndarray
                img2.append(train_img_resize)
                # 计算相似度 相关性
                img1_row = img1.reshape(-1)
                img2_row = img2[j].reshape(-1)
                # 长度不同就补零
                if len(img1_row) < len(img2_row):
                    zero_vector = torch.zeros( len(img2_row) - len(img1_row))
                    img1_row_new=torch.cat((img1_row,zero_vector), 0)
                    temp = np.corrcoef(img1_row_new, img2_row)
                    correlation[i].append(temp[0][1])
                else:
                    zero_vector = torch.zeros(len(img1_row) - len(img2_row))
                    img2_row_new = torch.cat((img2_row, zero_vector), 0)
                    temp = np.corrcoef(img1_row, img2_row_new)
                    correlation[i].append(temp[0][1])
            temp_index = correlation[i].index(max(correlation[i]))
            index_of_max_value.append(temp_index)
            best_train_label = train_label[temp_index]
            if test_label == best_train_label:
                right_num = right_num+1
            print(i)
    acc.append(right_num*1.0/(10))         #######
    print(acc)


    # 训练集中 与第一张测试图片CC最高的图像
    # 测试集中的图片
    # plt.figure()
    # plt.imshow(img1)
    # plt.show()
    # plt.figure()
    # plt.show()

    # plt.imshow(img2[index_of_max_value[i]])

# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 20,
#          }
# plt.xlabel('R', font1)
# plt.ylabel('CC', font1)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.plot(np.arange(0, j+1, 1), correlation[i], color=(0, 0.5, 1), linewidth=1.5)
# plt.grid(True)
# plt.xticks(np.arange(0, j, 500))
# plt.yticks(np.arange(0, 1 + 0.05, 0.05))
# plt.xlim([s,2])
