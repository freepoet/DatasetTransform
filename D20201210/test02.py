# -*- coding: utf-8 -*-
"""
@File    : test02.py
@Time    : 12/10/20 10:50 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 10:01 AM      1.0         None
# @Software: PyCharm
"""
import torch
from mlp import MLP
from torch import optim
from torch import nn
model = MLP(28*28,300,200,10)
print(model)

optimizer=optim.SGD(params = model.parameters(),lr=0.01)
data=torch.randn(10,28*28)
output=model(data)
label=torch.Tensor([1,2,3,4,5,6,7,8,9,0]).long()
criterion=nn.CrossEntropyLoss()
loss=criterion(output,label)
print(loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
