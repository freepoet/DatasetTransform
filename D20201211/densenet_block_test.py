# -*- coding: utf-8 -*-
"""
@File    : densenet_block_test.py
@Time    : 12/11/20 10:37 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 10:37 PM      1.0         None
# @Software: PyCharm
"""
import torch
from densenet_block import DenseBlock
densenet_model=DenseBlock(64,32,6).cuda()
print(densenet_model)
input=torch.randn(1,64,32,32).cuda()
output=densenet_model(input)
print(output.shape)
