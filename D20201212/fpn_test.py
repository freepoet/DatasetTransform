# -*- coding: utf-8 -*-
"""
@File    : fpn_test.py
@Time    : 12/12/20 9:40 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/12/20 9:40 PM      1.0         None
# @Software: PyCharm
"""
import torch
from fpn import FPN
net=FPN([3,4,6,3]).cuda()
print(net)
input=torch.randn(1,3,224,224).cuda()
output=net(input)
