# -*- coding: utf-8 -*-
"""
@File    : inceptionv1_test.py
@Time    : 12/11/20 4:39 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 4:39 PM      1.0         None
# @Software: PyCharm
"""
import torch
from inceptionv1 import Inceptionv1
net=Inceptionv1(3,64,32,64,64,96,32).cuda()
input=torch.randn(1,3,32,32).cuda()
output=net(input)