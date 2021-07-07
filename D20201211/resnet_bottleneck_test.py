# -*- coding: utf-8 -*-
"""
@File    : resnet_bottleneck_test.py
@Time    : 12/11/20 9:09 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 9:09 PM      1.0         None
# @Software: PyCharm
"""
import torch
from resnet_bottleneck import BottleNeck
bottleneck1_1=BottleNeck(64,256)
print(bottleneck1_1)
input=torch.randn(1,64,32,32)
out=bottleneck1_1(input)
print(input.shape)
print(out.shape)