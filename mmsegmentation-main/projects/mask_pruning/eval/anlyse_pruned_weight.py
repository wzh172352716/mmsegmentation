import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Read coordinate data from text file
data = np.loadtxt('result_pruned_weight.txt')  # 请将文件路径替换为你的文件路径

# Extract X, Y and Z coordinate data separately
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

xi = np.linspace(min(x), max(x), 500)
yi = np.linspace(min(y), max(y), 500)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x, y), z, (xi, yi), method='linear')

# Create 3D surface plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(xi, yi, zi, cmap='viridis')
#ax.plot_trisurf(x, y, z, cmap='viridis')

# set label 
ax.set_xlabel('lr_rate')
ax.set_ylabel('mask_factor')
ax.set_zlabel('number of the pruned weight')
fig.colorbar(surface, shrink=0.5, aspect=10,pad=0.2)
ax.view_init(elev=20, azim=80)

#plt.savefig('/home/wang/work/test/modelpruningsequentiallylearnablemasks/result/result_weight_1.png')

# 显示图形
plt.show()