from svm import SVM, RBF_kernel
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_data = np.loadtxt(f"data5/training_3.txt")
    gamma = 100
    plt.figure(3)

    print(f"-------------training_data3, gamma = {gamma}---------------")
    svm = SVM(training_data, kernel=RBF_kernel)
    svm.train()
    svm.plot_contour()
    plt.title(r"$\gamma$ = {}".format(gamma))
    plt.legend(['y = 1', 'y = -1'])

    plt.show()

