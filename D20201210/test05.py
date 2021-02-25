# -*- coding: utf-8 -*-
"""
@File    : test05.py
@Time    : 12/10/20 10:52 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 10:52 PM      1.0         None
# @Software: PyCharm
"""
import torch
import torch.nn.functional as F
from torch import nn
conv=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True)
# print(conv)
# print(conv.weight.shape)
input = torch.randn(1,1,2,2)
print(input)
# ouput=conv(input)
activation1=nn.Sigmoid()
output=activation1(input)
# print(output)
activation2=nn.ReLU(inplace=True)
activation2(input)
# print(input)
score=torch.randn(1,4)
print(score)
out=F.softmax(score,1)
print(out)
