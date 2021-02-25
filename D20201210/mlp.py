# -*- coding: utf-8 -*-
"""
@File    : mlp.py
@Time    : 12/10/20 10:10 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 10:10 AM      1.0         None
# @Software: PyCharm
"""
from torch import nn
class MLP(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim):
        super(MLP,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(in_dim,hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2,out_dim),
            nn.ReLU()
        )
    def forward(self,x):
        x=self.layer(x)
        return x

