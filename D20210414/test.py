import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# 设置距离
x = np.array([0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10])

# 设置相似度
y = np.array([0.8579087793827057, 0.8079087793827057, 0.7679087793827057, 0.679087793827057,
              0.5579087793827057, 0.4579087793827057, 0.3079087793827057, 0.3009087793827057,
              0.2579087793827057, 0.2009087793827057, 0.1999087793827057, 0.1579087793827057,
              0.0099087793827057, 0.0079087793827057, 0.0069087793827057, 0.0019087793827057,
              0.0000087793827057])
# 实现函数
func = interpolate.interp1d(x, y, kind='cubic')
# 插值法之后的x轴值，表示从0到10间距为0.5的200个数
xnew = np.arange(0, 10, 0.05)
# 利用xnew和func函数生成ynew,xnew数量等于ynew数量
ynew = func(xnew)

# 原始折线
plt.plot(x, y, "r", linewidth=1)

# 平滑处理后曲线
plt.plot(xnew, ynew)
# 设置x,y轴代表意思
plt.xlabel("The distance between POI  and user(km)")
plt.ylabel("probability")
# 设置标题
plt.title("The content similarity of different distance")
# 设置x,y轴的坐标范围
plt.xlim(0, 10, 8)
plt.ylim(0, 1)

plt.show()
