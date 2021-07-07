# -*- coding: utf-8 -*-
"""
@File    : inceptionv2_test.py
@Time    : 12/11/20 5:31 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 5:31 PM      1.0         None
# @Software: PyCharm
"""
import torch
from inceptionv2 import Inceptionv2
net=Inceptionv2(192,96,48,64,64,96,96,64).cuda()
print(net)
input=torch.randn(1,192,32,32).cuda()
output=net(input)
print(input.shape)
print(output.shape)

