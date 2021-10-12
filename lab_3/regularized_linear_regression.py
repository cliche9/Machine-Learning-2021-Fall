import numpy as np
import matplotlib.pyplot as plt

class RegularizedLinearRegression:

    def __init__(self, x, y, lr=0.3, r_lambda=0):
        self.x = x
        self.y = y
        self.lr = lr
        self.r_lambda = r_lambda
        self.expand = lambda x, k: np.array([x ** i for i in range(2, k+1)]).T

    def train(self):
        # expand x
        m = self.x.shape[0]
        x_t = np.column_stack( (np.ones(m), self.x) )
        self.x = np.concatenate( (x_t, self.expand(self.x, 5)), axis=1 )
        x_t = self.x
        # Matrix following lambda
        n = x_t.shape[1]
        L = np.identity(n)
        L[0][0] = 0
        # solve theta
        theta = np.dot( np.linalg.inv( np.dot(x_t.T, x_t) + self.r_lambda * L ), np.dot( x_t.T, self.y ) )

        return theta

    def plot_fit_curve(self, theta):
        x_hat = np.arange(np.min(self.x[:, 1]), np.max(self.x[:, 1]), 0.01)
        x = np.column_stack( (np.ones(len(x_hat)), x_hat) )
        x = np.concatenate( (x, self.expand(x_hat, 5)), axis=1 )
        y_hat = np.dot(x, theta)
        plt.plot(x_hat, y_hat, '--', label=f'Lambda={self.r_lambda}')

if __name__ == "__main__":
    x = np.loadtxt("data3/ex3Linx.dat")
    y = np.loadtxt("data3/ex3Liny.dat")
    plt.plot(x, y, 'o', label='Training data')
    for r_lambda in {0, 1, 10}:
        regLinear = RegularizedLinearRegression(x, y, r_lambda=r_lambda)
        theta = regLinear.train()
        print(theta)
        regLinear.plot_fit_curve(theta)

    plt.title("Regularized Linear Regression")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

