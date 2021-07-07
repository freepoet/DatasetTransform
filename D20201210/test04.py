# -*- coding: utf-8 -*-
"""
@File    : test04.py
@Time    : 12/10/20 4:42 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/10/20 4:42 PM      1.0         None
# @Software: PyCharm
"""
import torch
from torchvision import models
from tensorboardX import SummaryWriter
a=torch.randn(3,3)
b=models.vgg16()
if torch.cuda.is_available():
    a=a.cuda()
    b=b.cuda()
device=torch.device('cuda:0')
c=torch.randn(3,3,device=device,requires_grad=True)

writer=SummaryWriter()
for i in range(10):
    writer.add_scalar('quadratic',i**2,global_step=i)
    writer.add_scalar('exponential', 2 ** i, global_step=i)





