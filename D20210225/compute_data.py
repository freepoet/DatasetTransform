# -*- coding: utf-8 -*-
"""
@File    : compute_data.py
@Time    : 30/09/2020 18:46
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
30/09/2020 18:46      1.0         None
# @Software: PyCharm
"""
import numpy as np
area_ratios = np.load('/home/n/Github/Myrepo/CaptureTargetInImage/20200930/area_ratios.npy')
width_ratios = np.load("/home/n/Github/Myrepo/CaptureTargetInImage/20200930//width_ratios.npy")
height_ratios = np.load("/home/n/Github/Myrepo/CaptureTargetInImage/20200930//height_ratios.npy")
area_ratios=area_ratios.tolist()
width_ratios=width_ratios.tolist()
height_ratios=height_ratios.tolist()
area_ratios_mean=np.median(area_ratios)
print(area_ratios_mean)
width_ratios_mean=np.median(width_ratios)
print(width_ratios_mean)
height_ratios_mean=np.median(height_ratios)
print(height_ratios_mean)
print(max(area_ratios))
big_value_index = [k for k,v in enumerate(area_ratios) if v<=0.2]

print(len(big_value_index)/len(area_ratios))
