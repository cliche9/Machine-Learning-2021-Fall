import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import linear_regression as lr

''' Getting best theta_0 and theta_1 '''
x = np.loadtxt("data1/ex1_1x.dat").reshape(-1, 1)
y = np.loadtxt("data1/ex1_1y.dat").reshape(-1, 1)
theta, loop_count = lr.linear_regression(x, y, 0.07, 3000)
m = len(x)
x = np.column_stack((np.ones(m), x))

''' Understanding J(theta) '''
theta_0_vals = np.linspace(-3, 3, 100)
theta_1_vals = np.linspace(-1, 1, 100)
'''
linspace vs. logspace
theta_0_vals = np.logspace(-3, 3, 100)
theta_1_vals = np.logspace(-1, 1, 100)
'''
J_vals = np.zeros((len(theta_0_vals), len(theta_1_vals)))

for i in range(len(theta_0_vals)):
    for j in range(len(theta_1_vals)):
        t = np.array([[theta_0_vals[i]], [theta_1_vals[j]]])
        J_vals[i][j] = np.power(np.dot(x, t) - y, 2).sum() / (2 * m)

J_vals = J_vals.T
plt.figure(2)
# 数据范围
cNorm = colors.Normalize(vmin=np.min(J_vals), vmax=np.max(J_vals))
# 颜色范围
cmaps = 'rainbow'
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmaps)
# 绘制3D
plt3d = plt.axes(projection='3d')
theta_0, theta_1 = np.meshgrid(theta_0_vals, theta_1_vals)
plt3d.plot_surface(theta_0, theta_1, J_vals, norm=cNorm, cmap=cmaps)
plt3d.set_xlabel(r'$\theta_0$')
plt3d.set_ylabel(r'$\theta_1$')
# 增加colorbar
plt.colorbar(scalarMap)

# 等高线图
plt.figure(3)
cmaps = cmx.hot
plt.contourf(theta_0, theta_1, J_vals, 15, alpha=0.75, cmap=cmaps, norm=cNorm)
C = plt.contour(theta_0, theta_1, J_vals, 15, colors='black')
plt.clabel(C, inline=True, fontsize=10)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
contourMap = cmx.ScalarMappable(norm=cNorm, cmap=cmaps)
plt.colorbar(contourMap)

plt.show()