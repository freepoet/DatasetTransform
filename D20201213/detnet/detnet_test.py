# -*- coding: utf-8 -*-
"""
@File    : detnet_test.py
@Time    : 12/13/20 9:53 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/13/20 9:53 AM      1.0         None
# @Software: PyCharm
"""
import torch
from detnet.detnet_bottleneck import DetBottleneck
bottleneck_b=DetBottleneck(1024,256,1,True).cuda()
bottleneck_a1=DetBottleneck(256,256,True).cuda()
bottleneck_a2=DetBottleneck(256,256,True).cuda()
input=torch.randn(1,1024,14,14).cuda()
output1=bottleneck_b(input)
output2=bottleneck_a1(output1)
output3=bottleneck_a2(output2)


