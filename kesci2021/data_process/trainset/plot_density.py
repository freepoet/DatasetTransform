import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import skewnorm

area_ratios = np.load('target_area_including_name.npy')
width_ratios = np.load('target_width_including_name.npy')
height_ratios = np.load('target_height_including_name.npy')

plt.figure(1)
font={'family':'serif', 'style':'italic', 'weight':'normal', 'color':'k', 'size':10 } #调用方式如下： plt.plot(x,y,fontdict=font)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(axis='both',which='major',labelsize=14)
my_x_ticks = np.arange(0,160, 10)
my_y_ticks = np.arange(0,160, 10)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('Width', fontdict=font, fontsize=15)
plt.ylabel('Height', fontdict=font, fontsize=15)
plt.xlim(0,160)
plt.ylim(0,160)

plt.grid(True)


# Create and shor the 2D Density plot
ax = sns.kdeplot(width_ratios, height_ratios, cmap="Reds", shade=False, bw=.15, cbar=True)
ax.set(xlabel='width', ylabel='height')

plt.show()