import linear_regression as lr
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("data1/ex1_2x.dat").reshape(-1, 2)
y = np.loadtxt("data1/ex1_2y.dat").reshape(-1, 1)
m = len(y)

# axis = 0: 沿列计算
# axis = 1: 沿行计算
sigma = np.std(x, axis=0)
mu = np.mean(x, axis=0)
# normalization
x[:, 0] = (x[:, 0] - mu[0]) / sigma[0]
x[:, 1] = (x[:, 1] - mu[1]) / sigma[1]
x_new = np.column_stack((np.ones(m), x))
# plot changes of J_theta by iterations
alpha = np.arange(0.1, 1.3, 0.1)
alpha_num = len(alpha)
loop_max = 10
theta = [np.zeros([len(x_new[0, :]), 1]) for i in range(alpha_num)]
J = np.zeros([loop_max, alpha_num])

for iteration_count in range(loop_max):
    for t in range(alpha_num):
        y_hat = np.dot(x_new, theta[t]).reshape(-1, 1)
        J[iteration_count][t] = np.power(y_hat - y, 2).sum() / (2 * m)
        theta[t] = theta[t] - 1 / m * alpha[t] * np.sum(x_new * (y_hat - y), axis=0).reshape(-1, 1)

plt.figure()
for t in range(alpha_num):
    plt.plot(np.arange(loop_max), J[:, t], '-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.legend([r'$\alpha$ = ' + str(round(t, 2)) for t in alpha])

theta_end, loop_count = lr.linear_regression(x, y, 0.96, 400)
print('Loop count = %d' % loop_count)
print('Final values of theta:')
for i in range(3):
    print('theta_' + str(i) + ' = %f' % theta_end[i])
x_predict = np.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]]).reshape(1, -1)
print('Price of a house with 1650 square feet and 3 bedrooms = %f' % (np.dot(x_predict, theta_end)))

plt.show()



