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
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
show=ToPILImage()
from .test02 import calculate

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
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
for i, test_img in enumerate(testloader, 0):
    for j,  train_img in enumerate(trainloader, 0):
        print((data[0]))
        show((data[0]+1)/2)
        # plt.imshow(data[0])
        # plt.show
        break