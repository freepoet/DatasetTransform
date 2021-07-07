# -*- coding: utf-8 -*-
"""
@File    : inceptionv1.py
@Time    : 12/11/20 3:49 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/11/20 3:49 PM      1.0         None
# @Software: PyCharm
"""
import torch
from torch import nn
import torch.nn.functional as F
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding=0):
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding)
    def forward(self,x):
        x=self.conv(x)
        return F.relu(x,inplace=True)

class Inceptionv1(nn.Module):
    def __init__(self,in_dim,hid1_1,hid2_1,hid2_3,hid3_1,hid3_5,hid4_1):
        super(Inceptionv1,self).__init__()
        self.branch1x1 = BasicConv2d(in_dim,hid1_1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_dim, hid2_1, 1),
            BasicConv2d(hid2_1, hid2_3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_dim, hid3_1,  kernel_size=1),
            BasicConv2d(hid3_1, hid3_5,  kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            BasicConv2d(in_dim, hid4_1, 1)
        )

    def forward(self,x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        output = torch.cat((b1,b2,b3,b4),dim=1)
        return output

