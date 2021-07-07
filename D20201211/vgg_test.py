# -*- coding: utf-8 -*-
"""
@File    : vgg_test.py
@Time    : 12/11/20 3:09 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 3:09 PM      1.0         None
# @Software: PyCharm
"""
import torch
import torch.nn.functional as F
from vgg import VGG
net=VGG(5).cuda()
input=torch.randn(1,3,224,224).cuda()
scores=net(input)
pros=F.softmax(scores,1)
print(net.features)
print(net.classifier)
print(scores)
print(pros)

