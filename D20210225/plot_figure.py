# -*- coding: utf-8 -*-
""" """

import seaborn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# sns.set_palette("hls")
#   must absolute path!!!!!!!!!!!!!!!!!!!!!
area_ratios = np.load('C:/ning/Github/MyRepo/LearningNotes/D20210225/area_ratios.npy')
width_ratios = np.load("C:/ning/Github/MyRepo/LearningNotes/D20210225/width_ratios.npy")
height_ratios = np.load("C:/ning/Github/MyRepo/LearningNotes/D20210225/height_ratios.npy")
# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
# def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
#     plt.hist(myList, 3000)  # bins = 50，顺便可以控制bin宽度
#     plt.xlim(Xmin, Xmax)
#     plt.ylim(Ymin, Ymax)
#     plt.xlabel(Xlabel)  # 横轴名
#     plt.ylabel(Ylabel)  # 纵轴名
#     plt.title(Title)
# 按照固定区间长度绘制频率分布直方图
# bins_interval 区间的长度
# margin        设定的左边和右边空留的大小
# def probability_distribution(data, bins_interval, margin):
#     bins = np.arange(min(data), max(data) , bins_interval)
#     plt.xlim(min(data) - margin, max(data) + margin)
#     plt.title("Probability-distribution")
#     plt.xlabel('Interval')
#     plt.ylabel('Probability')
#     #频率分布normed=True，频次分布normed=False
#     plt.hist(x=data, bins=bins, histtype='bar')

# seaborn.set(style='white')
with seaborn.axes_style("dark"):
    seaborn.jointplot(x=width_ratios, y=height_ratios, kind="scatter",color='k',s=10).set_axis_labels("relative width", "relative height")

# plt.figure(2)
# temp=(area_ratios.tolist())
# probability_distribution(temp,0.001,0)
plt.show()




