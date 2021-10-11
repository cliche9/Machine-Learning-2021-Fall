import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    h_z = []
    for x in z:
        x = x.item()
        if x >= 0:
            h_z.append(1.0 / (1 + np.exp(-x)))
        else:
            h_z.append(np.exp(x) / (1 + np.exp(x)))
    return np.mat(h_z).transpose()

def init_data():
    x = np.loadtxt("data2/ex2x.dat")
    label = np.loadtxt("data2/ex2y.dat")
    # normalization
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x[:, 0] = (x[:, 0] - mu[0]) / sigma[0]
    x[:, 1] = (x[:, 1] - mu[1]) / sigma[1]
    x = np.insert(x, 0, 1, axis=1)
    predict_x = np.mat([ 1, (20 - mu[0]) / sigma[0], (80 - mu[1]) / sigma[1] ])
    return x, label, predict_x

def grad_descent(x, label):
    x = np.mat(x)
    label = np.mat(label).transpose()
    m, n = np.shape(x)
    # 初始化回归系数
    theta = np.zeros((n, 1))
    # learning rate
    alpha = 0.01
    loop_max = 200000
    loop_count = 0
    loss = 0
    gradient_loss_list = []

    for i in range(loop_max):
        loop_count += 1
        # sigmoid
        h = sigmoid(x * theta)
        old_loss = loss
        # 计算loss
        loss = 1 / m * (np.multiply((-label), np.log(h)) - np.multiply((1 - label), np.log(1 - h))).sum()
        gradient_loss_list.append(loss)
        if abs(loss - old_loss) < 1e-6:
            break
        # 梯度上升转为梯度下降
        grad = 1 / m * x.transpose() * (h - label)
        theta = theta - alpha * grad

    plt.plot(np.arange(loop_count), gradient_loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Gradient Loss by Iteration')
    plt.show()
    return theta, loop_count

def newton_method(x, label):
    x = np.mat(x)
    label = np.mat(label).transpose()
    m, n = np.shape(x)
    # 初始化theta
    thetas = np.zeros((n, 1))
    loop_max = 2000
    loop_count = 0
    loss = 0
    newton_loss_list = []

    for i in range(loop_max):
        loop_count += 1
        # sigmoid
        h = sigmoid(x * thetas)
        old_loss = loss
        # 计算loss
        loss = 1 / m * (np.multiply((-label), np.log(h)) - np.multiply((1 - label), np.log(1 - h))).sum()
        newton_loss_list.append(loss)
        if abs(loss - old_loss) < 1e-6:
            break
        # gradient
        grad = 1 / m * x.transpose() * (h - label)
        # Newton's Method
        hessian = 0
        h_array = np.asarray(h)
        for j in range(m):
            hessian = hessian + h_array[j][0] * (1 - h_array[j][0]) * x[j, :].transpose() * x[j, :]
        hessian /= m
        # 更新theta
        thetas = thetas - np.linalg.inv(hessian) * grad

    plt.plot(np.arange(loop_count), newton_loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Newton Loss by Iteration')
    plt.show()
    return thetas, loop_count


def plotFit(thetas, x, data_y):
    pos_index = np.where(x == 1)
    neg_index = np.where(data_y == 0)
    x1 = np.arange(min(x[:, 1]), max(x[:, 1]), 0.1)
    n = len(thetas)
    titles = ["Gradient Descent", "Newton's Method"]

    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.title(titles[i])
        plt.scatter(x[pos_index, 1], x[pos_index, 2], marker='+')
        plt.scatter(x[neg_index, 1], x[neg_index, 2], marker='o')
        plt.xlabel('x1')
        plt.ylabel('x2')
        x2 = ((-thetas[i][0, 0] - thetas[i][1, 0] * x1) / thetas[i][2, 0])
        plt.plot(x1, x2, 'red')
    plt.show()

if __name__ == '__main__':
    x, label, predict_x = init_data()
    thetas = []
    # gradient descent
    theta, loop_count = grad_descent(x, label)
    thetas.append(theta)
    print(f"Gradient descent result: thetas =\n{theta}\nLoop count = {loop_count}")
    predict_y = sigmoid(predict_x * theta).item()
    print(f"Gradient descent: Predict_y = {1 - predict_y}")
    # Newton's method
    theta, loop_count = newton_method(x, label)
    thetas.append(theta)
    print(f"Newton's method result: thetas =\n{theta}\nLoop count = {loop_count}")
    isAdmitted = ''
    predict_y = sigmoid(predict_x * theta).item()
    print(f"Newton's method: Predict_y = {1 - predict_y}")
    if (predict_y < 0.5):
        isAdmitted = 'not'
    print('Student with a score of 20 on Exam 1 and a score of 80 on Exam 2 will ' + isAdmitted + ' be admitted')

    plotFit(thetas, x, label)




