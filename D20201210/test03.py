# -*- coding: utf-8 -*-
"""
@File    : test03.py
@Time    : 12/10/20 4:06 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 4:06 PM      1.0         None
# @Software: PyCharm
"""
import torch
from torch import nn
from torchvision import models
# vgg=models.vgg16(pretrained=True)
vgg=models.vgg16()
# print(len(vgg.features))
# print(len(vgg.classifier))
# print(vgg.classifier[-1])
state_dict=torch.load("vgg_pretrainedz                  .pth")
t=vgg.state_dict()
n=state_dict.items()
vgg.load_state_dict({k:v for k,v in n if k in t})
