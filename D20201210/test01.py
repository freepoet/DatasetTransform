# -*- coding: utf-8 -*-
"""
@File    : test01.py
@Time    : 12/10/20 9:54 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 9:54 AM      1.0         None
# @Software: PyCharm
"""
import torch
from torch import nn
from torch.autograd import Variable
from perception_sequential import Perception
model=Perception(100,1000,2).cuda()
print(model)
input=torch.randn(2,100).cuda()
output=model(input)
print(output)
label=Variable(torch.Tensor([-0.5,0])).long().cuda()
criterion=nn.CrossEntropyLoss()
loss_nn=criterion(output,label)
print(loss_nn)