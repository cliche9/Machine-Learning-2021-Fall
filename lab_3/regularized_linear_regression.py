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
    plt.figure(1)
    plt.plot(x, y, 'o', label='Training data')
    lambdas = [0, 1, 3, 5, 7, 10, 50]
    thetas = []
    for r_lambda in lambdas:
        regLinear = RegularizedLinearRegression(x, y, r_lambda=r_lambda)
        theta = regLinear.train()
        thetas.append(theta)
        print(theta)
        regLinear.plot_fit_curve(theta)

    plt.title("Regularized Linear Regression")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    # l2_norm changes by lambda
    l2_norms = [np.linalg.norm(theta) for theta in thetas]
    plt.figure(2)
    plt.plot(lambdas, l2_norms, 'o')
    plt.title(r"L2_Norm by $\lambda$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("L2-Norm")
    # easy to observe
    plt.figure(3)
    plt.plot(lambdas[1:], l2_norms[1:])
    plt.title(r"L2_Norm by $\lambda$")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("L2-Norm")
    plt.show()

