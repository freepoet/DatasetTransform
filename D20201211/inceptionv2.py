# -*- coding: utf-8 -*-
"""
@File    : inceptionv2.py
@Time    : 12/11/20 4:49 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 4:49 PM      1.0         None
# @Software: PyCharm
"""
import torch
from torch import nn
import torch.nn.functional as F
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding=0):
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding)
        self.bn=nn.BatchNorm2d(out_channels,eps=1e-5)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x,inplace=True)

class Inceptionv2(nn.Module):
    def __init__(self,in_dim,hid1_1,hid2_1,hid2_2,hid3_1,hid3_2,hid3_3,hid4_1):
        super(Inceptionv2,self).__init__()
        self.branch1 = BasicConv2d(in_dim,hid1_1,1,padding=0)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_dim,hid2_1, 1, padding=0),
            BasicConv2d(hid2_1, hid2_2, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_dim, hid3_1, 1, padding=0),
            BasicConv2d(hid3_1, hid3_2, 3, padding=1),
            BasicConv2d(hid3_2, hid3_3, 3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1,count_include_pad=False),
            BasicConv2d(in_dim, hid4_1, 1, padding=0)
        )
    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out=torch.cat((b1,b2,b3,b4),dim=1)
        return out

