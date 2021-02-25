# -*- coding: utf-8 -*-
"""
@File    : resnet_bottleneck.py
@Time    : 12/11/20 8:29 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 8:29 PM      1.0         None
# @Software: PyCharm
"""
import  torch.nn as nn
class BottleNeck(nn.Module):
    def __init__(self,in_dim,out_dim,stride=1):
        super(BottleNeck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1,bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_dim, in_dim, 3, stride, 1,bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_dim, out_dim, 1,bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.relu=nn.ReLU(inplace=True)
        #Downsample
        self.downsample=nn.Sequential(
            nn.Conv2d(in_dim,out_dim,1),
            nn.BatchNorm2d(out_dim),
        )
    def forward(self,x):
        identity=x
        out=self.bottleneck(x)
        identity=self.downsample(x)
        out+=identity
        out=self.relu(out)
        return out
