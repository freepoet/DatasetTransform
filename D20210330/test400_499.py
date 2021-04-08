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
gpu_id = 4
#读取MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-255  转为  0-1
    transforms.Normalize((0.5,),(0.5,))  # 0-1  转为  -1-1
])
trainset=torchvision.datasets.MNIST(
    root='../../../data/',
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
    root='../../../data/',
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
# 尺度变化
start = 0.5
k_int=0
for k in np.arange(start, 2, 0.1):
    right_num = 0
    acc.append([])
        # k = round(k, 2)  # 保留两位小数
    for i, test_data in enumerate(testloader, 0):
        if i >= 400 and i <= 499: # 仅仅测试20张图片
            # 二维list
            # 原始图片为NCWH的Tensor 压缩成WH的Tensor
            test_img = (test_data[0].squeeze(0)).squeeze(0)
            test_label = test_data[1]
            # 获取测试图片的宽 高
            width = test_img.shape[0]
            height = test_img.shape[1]
            #tenor - cuda
            img1 = test_img.cuda(gpu_id)
            img2 = []
            train_label = []
            similarity = []
            for j, train_data in enumerate(trainloader, 0):
                if j < 50000:
                    # 原始图片为NCWH的Tensor 压缩成WH的Tensor
                    train_img = (train_data[0].squeeze(0)).squeeze(0)
                    train_label.append(train_data[1])
                    # 训练图片resize成测试图片一样的大小 将训练图片转换为PIL
                    train_img_pil = transforms.ToPILImage()(train_img)
                    # 按照拉伸比例 resize图片
                    new_width = width
                    new_height = int(round(height*k))
                    train_img_resize = train_img_pil.resize((new_width, new_height), Image.ANTIALIAS) #PIL 格式图片才能resize
                    # PIL转tensor 3维转2维
                    train_img_resize = transforms.ToTensor()(train_img_resize)
                    train_img_resize.squeeze_()
                    # Tensor cuda
                    img2.append(train_img_resize.cuda(gpu_id))
                    # 计算相似度 相关性
                    img1_row = img1.reshape(-1)
                    img2_row = img2[j].reshape(-1)
                    # 长度不同就补零
                    if len(img1_row) < len(img2_row):
                        # zero_vector = torch.zeros( len(img2_row) - len(img1_row)).cuda(0)
                        # img1_row_new=torch.cat((img1_row,zero_vector), 0)
                        # temp = torch.cosine_similarity(img1_row_new, img2_row, dim=0)
                        # similarity.append(temp)
                        temp= torch.cosine_similarity(img1_row, img2_row[:len(img1_row)], dim=0)
                        similarity.append(temp)

                    else:
                        zero_vector = torch.zeros(len(img1_row) - len(img2_row)).cuda(gpu_id)
                        img2_row_new = torch.cat((img2_row, zero_vector), 0)
                        temp = torch.cosine_similarity(img1_row, img2_row_new, dim=0)
                        similarity.append(temp)
                    print(k_int, i, j)
            # max index
            temp_index = similarity.index(max(similarity))
            index_of_max_value.append(temp_index)
            best_train_label = train_label[temp_index]
            if test_label == best_train_label:
                right_num = right_num + 1
    acc[k_int].append(right_num*1.0/(100))
    print('ACC:'+str(acc[k_int]))
    k_int=k_int+1


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
