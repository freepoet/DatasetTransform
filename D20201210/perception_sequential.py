# -*- coding: utf-8 -*-
"""
@File    : perception_sequential.py
@Time    : 12/10/20 9:44 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 9:44 AM      1.0         None
# @Software: PyCharm
"""
from torch import nn
class Perception(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(Perception,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(in_dim,hid_dim),
            nn.Sigmoid(),
            nn.Linear(hid_dim,out_dim),
            nn.Sigmoid()
        )
    def forward(self,x):
        y=self.layer(x)
        return y
