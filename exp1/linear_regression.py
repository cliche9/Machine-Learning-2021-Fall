import numpy as np

''' nD Linear Regression '''
def linear_regression(x, y, alpha, loop_max):
    # x, y行数
    m = len(y)
    # 对x增加一行1, 实现 + theta0
    x = np.column_stack((np.ones(m), x))
    # 获取x列数
    n = len(x[0])
    # 初始化theta, 为n * 1矩阵
    theta = np.zeros((n, 1))
    loop_count = 0
    # 批量梯度下降算法公式：
    # theta = theta - 1 / m * alpha * ∑( (y-y_hat) * x )
    # gradient descent
    for i in range(loop_max):
        loop_count += 1
        # x(m * 2) * theta(2 * 1)
        y_hat = np.dot(x, theta)
        # 记录旧的theta
        theta_old = theta
        # 梯度下降求解新theta
        theta = theta - 1 / m * alpha * np.sum(x * (y_hat - y), axis=0).reshape(-1, 1)
        # theta变化极小
        if (abs(theta - theta_old) < 1e-6).all():
            break
    
    return theta, loop_count