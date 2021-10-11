import numpy as np
import matplotlib.pyplot as plt
import linear_regression as lr

x = np.loadtxt("data1/ex1_1x.dat").reshape(-1, 1)
y = np.loadtxt("data1/ex1_1y.dat").reshape(-1, 1)

plt.figure()
plt.scatter(x, y, marker='o', label='True data')
plt.xlabel('Age in years')
plt.ylabel('Height in meters')

# 求解theta和loop次数
theta, loop_count = lr.linear_regression(x, y, 0.07, 3000)
print('Loop count = %d' % loop_count)
# 加一列"1"作为 theta_0的乘数
x = np.column_stack((np.ones(len(x)), x))
y_hat = np.dot(x, theta)

plt.plot(x[:, 1], y_hat, '-r', label='Linear regression')
plt.legend()
# 预测3.5, 7
print('Age = 3.5  Height = %f\nAge = 7  Height = %f' % (theta[0] + 3.5 * theta[1], theta[0] + 7 * theta[1]))

plt.show()
