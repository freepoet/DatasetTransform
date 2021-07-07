import numpy as np
import matplotlib.pyplot as plt
traintarget_area= np.load('train_area_ratios.npy')
traintarget_width = np.load('train_width_ratios.npy')
traintarget_height = np.load('train_height_ratios.npy')


trainimg_area= np.load('trainimg_area.npy')
trainimg_width = np.load('trainimg_width.npy')
trainimg_height = np.load('trainimg_height.npy')

plt.figure(1)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.title('Squares',fontsize=24)
plt.tick_params(axis='both',which='major',labelsize=14)
font={'family':'serif', 'style':'italic', 'weight':'normal', 'color':'k', 'size':16 } #调用方式如下： plt.plot(x,y,fontdict=font)
plt.xlabel('Width', fontdict=font, fontsize=20)
plt.ylabel('Height', fontdict=font, fontsize=20)

plt.scatter(traintarget_width,traintarget_height,c='k')
# my_x_ticks = np.arange(0,1.1, 0.1)
# my_y_ticks = np.arange(0,1.1, 0.1)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)


img_scale=[640,640]
k=img_scale[0]/img_scale[1]

new_traintarget_height=[]
new_traintarget_width=[]
for i in np.arange(len(trainimg_height)):
    k=trainimg_height[i]/640
    new_traintarget_width.append(traintarget_width[i]/k)
    new_traintarget_height.append(traintarget_height[i]/k)
plt.figure(2)

font={'family':'serif', 'style':'italic', 'weight':'normal', 'color':'k', 'size':16 } #调用方式如下： plt.plot(x,y,fontdict=font)
plt.xlabel('Width', fontdict=font, fontsize=20)
plt.ylabel('Height', fontdict=font, fontsize=20)

plt.scatter(new_traintarget_width,new_traintarget_height,c='k')
# my_x_ticks = np.arange(0,1.1, 0.1)
# my_y_ticks = np.arange(0,1.1, 0.1)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
plt.show()

