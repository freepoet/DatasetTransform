# -*- coding: utf-8 -*-
"""
@File    : densenet_block.py
@Time    : 12/11/20 9:59 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 9:59 PM      1.0         None
# @Software: PyCharm
"""
import torch
from torch import nn
class BottleNeck(nn.Module):
    def __init__(self,n_channels,growth_rate):
        super(BottleNeck,self).__init__()
        Channels=4*growth_rate
        self.bottleneck=nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels,Channels,1,bias=False),
            nn.BatchNorm2d(Channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Channels, growth_rate, 3,padding=1, bias=False)
    )
    def forward(self,x):
        out=self.bottleneck(x)
        out=torch.cat((x,out),1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, n_channels, growth_rate,n_DenseBlocks):
        super(DenseBlock, self).__init__()
        layers=[]
        for i in range(n_DenseBlocks):
            layers.append(BottleNeck(n_channels+i*growth_rate,growth_rate))
        self.denseblock=nn.Sequential(*layers)
    def forward(self, x):
        out=self.denseblock(x)
        return out



