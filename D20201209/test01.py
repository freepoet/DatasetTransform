# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 12/9/20 10:26 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/9/20 10:26 AM      1.0         None
# @Software: PyCharm
"""
import torch
from torch import nn
x=torch.randn(1)
w=torch.ones(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)
print(x.is_leaf,w.is_leaf,b.is_leaf)
print(x.requires_grad,w.requires_grad,b.requires_grad)
y=w*x
z=y+b
print(y.is_leaf,z.is_leaf)
print(y.grad_fn,z.grad_fn)
z.backward()
print(w.grad,b.grad)
