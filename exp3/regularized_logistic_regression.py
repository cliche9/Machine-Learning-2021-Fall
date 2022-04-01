import numpy as np
import matplotlib.pyplot as plt
from map_feature import map_feature

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

class RegularizedLogisticRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def Newton(self, r_lambda=0):
        m, n = self.x.shape
        theta = np.zeros((n, 1))
        # matrix L
        L = np.identity(n)
        L[0][0] = 0
        # loop
        loop_max = 100
        loop = 0
        pre_loss = 0
        loss_list = []

        print(f'---------------------Lambda={r_lambda}-------------------')
        for i in range(loop_max):
            h = Sigmoid(np.dot(self.x, theta))
            # loss
            loss = -1/m * np.sum( (self.y * np.log(h) + (1 - self.y) * np.log(1 - h)) ) + r_lambda/(2 * m) * np.sum(theta[1:] ** 2)
            print(f'Loss = {loss}')
            loss_list.append(loss)
            if abs(loss - pre_loss) < 1e-6:
                break
            pre_loss = loss
            # gradient: 28 * 1
            grad = 1/m * np.dot(self.x.T, (h - self.y))
            for i in range(1, n):
                grad[i] += r_lambda/m * theta[i]
            # update theta
            # hessain: 28 * 28
            hessian = 0
            for i in range(m):
                hessian += h[i] * (1 - h[i]) * np.dot(self.x.T[:, i].reshape(-1, 1), self.x[i, :].reshape(1, -1))
            hessian /= m
            hessian += r_lambda/m * L
            # theta: 28 * 1
            theta -= np.dot(np.linalg.inv(hessian), grad)
            loop += 1

        return theta

if __name__ == "__main__":
    # load and scatter
    x = np.loadtxt("data3/ex3Logx.dat", delimiter=',')
    y = np.loadtxt("data3/ex3Logy.dat", delimiter=',').reshape(-1, 1)
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    # train
    regLogistic = RegularizedLogisticRegression(map_feature(x[:, 0], x[:, 1]), y)
    lambdas = [0, 1, 3, 5, 7, 10]
    thetas = []
    # boundary
    plt.figure(1)
    rows = cols = int(np.sqrt(len(lambdas)))
    if rows * cols < len(lambdas):
        cols += 1
    for k in range(len(lambdas)):
        plt.subplot(rows, cols, k + 1)
        plt.scatter(x[pos, 0], x[pos, 1], marker='o')
        plt.scatter(x[neg, 0], x[neg, 1], marker='+')
        # solve theta
        theta = regLogistic.Newton(r_lambda=lambdas[k])
        thetas.append(theta)
        # plot boundary
        u = np.linspace(-1, 1.5, 200)
        v = np.linspace(-1, 1.5, 200)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(map_feature(u[i], v[j]), theta)

        plt.contour(u, v, z.T, [0])
        plt.title(f'$\lambda$={lambdas[k]}')
        plt.xlabel('u')
        plt.ylabel('v')
    # lambda affects results
    plt.figure(2)
    l2_norms = [np.linalg.norm(theta) for theta in thetas]
    plt.plot(lambdas, l2_norms, 'o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('L2-Norm')
    # easy to observe
    plt.figure(3)
    plt.plot(lambdas[1:], l2_norms[1:])
    plt.xlabel(r'$\lambda$')
    plt.ylabel('L2-Norm')
    plt.show()