import numpy as np
import matplotlib.pyplot as plt
area_ratios = np.load('C:/ning/Github/MyRepo/LearningNotes/D20210225/area_ratios.npy')
width_ratios = np.load('C:/ning/Github/MyRepo/LearningNotes/D20210225/width_ratios.npy')
height_ratios = np.load('C:/ning/Github/MyRepo/LearningNotes/D20210225/height_ratios.npy')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.title('Squares',fontsize=24)
plt.tick_params(axis='both',which='major',labelsize=14)
font={'family':'serif', 'style':'italic', 'weight':'normal', 'color':'k', 'size':16 } #调用方式如下： plt.plot(x,y,fontdict=font)

plt.xlabel('RW', fontdict=font, fontsize=20)
plt.ylabel('RH', fontdict=font, fontsize=20)
plt.scatter(width_ratios,height_ratios,c='k')
my_x_ticks = np.arange(0,1.1, 0.1)
my_y_ticks = np.arange(0,1.1, 0.1)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()