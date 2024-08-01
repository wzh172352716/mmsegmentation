import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 从文本文件读取坐标数据
data1 = np.loadtxt('./result.txt')  # 请将文件路径替换为你的文件路径

# 分别提取X、Y和Z坐标数据
x_mIoU = data1[:, 0]
y_mIoU = data1[:, 1]
z_mIoU = data1[:, 2]

data2 = np.loadtxt('result_pruned_weight.txt')  # 请将文件路径替换为你的文件路径

# Extract X, Y and Z coordinate data separately
x_pruned_weights = data2[:, 0]
y_pruned_weights = data2[:, 1]
z_pruned_weights = data2[:, 2]

x = x_mIoU
y = y_mIoU
z = 0.9 * z_mIoU + (1-0.9) * z_pruned_weights

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

xi = np.linspace(min(x), max(x), 500)
yi = np.linspace(min(y), max(y), 500)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x, y), z, (xi, yi), method='linear')

# 创建3D曲面图
# 创建3D曲面图


surface = ax.plot_surface(xi, yi, zi, cmap='viridis')

#ax.set_yscale('log')
#ax.set_xscale('log')

#ax.plot_trisurf(x, y, z, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('lr_rate')
ax.set_ylabel('mask_factor')
ax.set_zlabel('mIoU')
fig.colorbar(surface, shrink=0.5, aspect=10,pad=0.2)
ax.view_init(elev=20, azim=80)

#plt.savefig('./result_6.png')

# 显示图形
plt.show()