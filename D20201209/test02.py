# -*- coding: utf-8 -*-
"""
@File    : test02.py
@Time    : 12/9/20 7:56 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/9/20 7:56 PM      1.0         None
# @Software: PyCharm
"""
import torch
from perception import Perception
net=Perception(2,3,2)
# print(net)
# for name, parameter in net.named_parameters():
#     print(name,parameter)
data=torch.randn(4,2)
# print(data)
out = net(data)
print(out)

