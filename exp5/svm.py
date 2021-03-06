import cvxopt
from tqdm import tqdm, trange
from cvxopt import solvers, matrix
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from enum import Enum
from itertools import combinations

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynominal_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def RBF_kernel(x, y, gamma=1/60):
    return np.exp(-gamma * linalg.norm(x - y) ** 2)

class SVM(object):

    def __init__(self, training_data, test_data=None, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = float(C) if C is not None else C
        self.training_x = training_data[:, :-1]
        self.training_y = training_data[:, -1]
        if test_data is not None:
            self.test_x = test_data[:, :-1]
            self.test_y = test_data[:, -1]

    def train(self):
        sample_num, feature_num = self.training_x.shape
        x, y = self.training_x, self.training_y

        # Kernel Matrix
        K = np.zeros((sample_num, sample_num))
        print("----------Calculating Kernel Matrix...----------")

        for i in trange(sample_num):
                for j in range(sample_num):
                    K[i, j] = self.kernel(x[i], x[j])

        print("----------Done----------")
        """
        min { 1/2 alpha^T * P * alpha * K + q^T * alpha }
        st. 
            0 <= alpha <= C
            alpha * y = 0  
        """
        print("----------Constructing coefficient matrices...----------")
        P = matrix(np.outer(y, y) * K)
        q = matrix(np.ones(sample_num) * -1)
        A = matrix(y, (1, sample_num))
        b = matrix(0.0)

        if self.C is None or self.C == 0:
            # hard-margin
            # -alpha_i <= 0
            G = matrix(np.diag(np.ones(sample_num) * -1))
            h = matrix(np.zeros(sample_num))
        else:
            tmp1 = np.diag(np.ones(sample_num) * -1)
            tmp2 = np.identity(sample_num)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(sample_num)
            tmp2 = np.ones(sample_num) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        print("----------Done----------")
        print("----------Solving QP problem by qp_solvers...----------")
        # solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        print("----------Done----------")
        print("----------Calculating weights...----------")
        # Lagrange multipliers: [alpha_1, ... , alpha_m]
        alpha = np.ravel(solution['x'])

        # ???weight vector --> \omega: ?????????????????? alpha > 0 ?????????
        # Support vectors have non zero lagrange multipliers
        sv = alpha > 1e-9
        idx_of_sv = np.arange(len(alpha))[sv]
        # support vectors with non zero lagrange multipliers
        self.alpha = alpha[sv]
        # support vector's data x
        self.sv = x[sv]
        # support vector's label y
        self.sv_y = y[sv]

        if self.kernel == linear_kernel:
            self.w = np.zeros(feature_num)
            for i in range(len(self.alpha)):
                # linear_kernel????????????????????????, ???????????????feature space, ?????????feature space??????space??????
                self.w += self.alpha[i] * self.sv_y[i] * self.sv[i]
        else:
            # ??????linear_kernel??????feature map??????space??????, ???????????????\omega, ????????????Kernel Matrix???prediction
            self.w = None

        # Support vectors have non zero lagrange multipliers
        if self.C is not None and self.C > 0:
            sv = np.bitwise_and(alpha > 1e-9, alpha < self.C)
            idx_of_sv = np.arange(len(alpha))[sv]
            # support vectors with non zero lagrange multipliers
            self.alpha = alpha[sv]
            # support vector's data x
            self.sv = x[sv]
            # support vector's label y
            self.sv_y = y[sv]
        print(f"{len(self.alpha)} support vectors out of {sample_num} points")

        # ?????????support vector???b?????????, ??????????????????: alpha == C????????????????????????
        self.b = 0
        for i in range(len(self.alpha)):
            # b^* = y^i - sigma { alpha_j * y^j * K_ij }
            # ????????????support vector i??????????????????support vector j?????????(????????????????)
            self.b += self.sv_y[i]
            self.b -= np.sum(self.alpha * self.sv_y * K[idx_of_sv[i], sv])
        self.b /= len(self.alpha)

        print("----------Done----------")

    def project(self, x):
        # ??????x???????????????test_sample
        if self.w is not None:
            # w??????, kernel function???linear_kernel, ????????????
            return np.dot(x, self.w) + self.b
        else:
            # w is None ----> non linear kernel, ?????????kernel matrix??????
            y_predict = np.zeros(len(x))
            for i in range(len(x)):
                s = 0
                for alpha, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                    s += alpha * sv_y * self.kernel(x[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, x):
        return np.sign(self.project(x))

    def plot_margin(self):
        def f(x, w, b, c=0):
            # ???????????????x, ???????????????y???, (x, y)???margin line???
            # w * (x, y) + b = c
            return (-w[0] * x - b + c) / w[1]
        # split training data by label
        training_plus_y = self.training_x[self.training_y == 1]
        training_minus_y = self.training_x[self.training_y == -1]
        # y == 1???training points
        plt.plot(training_plus_y[:, 0], training_plus_y[:, 1], 'ro')
        # y == -1???training points
        plt.plot(training_minus_y[:, 0], training_minus_y[:, 1], 'bo')
        # ??????support vectors
        plt.legend(['y = 1', 'y = -1'])
        plt.scatter(self.sv[:, 0], self.sv[:, 1], s=100, c='g')

        xmin = np.min(self.training_x)
        xmax = np.max(self.training_x)
        # ???????????????
        # w * x + b = 0
        x1 = xmin; y1 = f(x1, self.w, self.b)
        x2 = xmax; y2 = f(x2, self.w, self.b)
        plt.plot([x1, x2], [y1, y2], '-')

        # w * x + b = -1
        y1 = f(x1, self.w, self.b, -1)
        y2 = f(x2, self.w, self.b, -1)
        plt.plot([x1, x2], [y1, y2], 'g--')

        # w * x + b = 1
        y1 = f(x1, self.w, self.b, 1)
        y2 = f(x2, self.w, self.b, 1)
        plt.plot([x1, x2], [y1, y2], 'g--')
        plt.title('C = {:.1f}'.format(self.C))

    def plot_contour(self):
        # split training data by label
        training_plus_y = self.training_x[self.training_y == 1]
        training_minus_y = self.training_x[self.training_y == -1]
        # training points
        plt.plot(training_plus_y[:, 0], training_plus_y[:, 1], 'ro')
        plt.plot(training_minus_y[:, 0], training_minus_y[:, 1], 'bo')
        plt.legend(['y = 1', 'y = -1'])
        # support vectors
        plt.scatter(self.sv[:, 0], self.sv[:, 1], s=100, c='g')
        xmin = np.min(self.training_x)
        xmax = np.max(self.training_x)
        x1, x2 = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(xmin, xmax, 100))
        x = np.array([ [x1, x2] for x1, x2 in zip(np.ravel(x1), np.ravel(x2)) ])
        z = self.project(x).reshape(x1.shape)
        # plt.contour????????????
        plt.contour(x1, x2, z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(x1, x2, z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(x1, x2, z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    def test(self):
        y_predict = (self.predict(self.test_x))
        correct = np.sum(y_predict == self.test_y)
        self.wrong_list = np.where(y_predict != self.test_y)[0]
        print(f"{correct} out of {len(y_predict)} predictions correct, {len(y_predict) - correct} out of {len(y_predict)} predictions wrong")
        print(f"correct rate = {correct / len(y_predict)}\nwrong rate = {(len(y_predict) - correct) / len(y_predict)}")

    def training_error(self):
        y_predict = (self.predict(self.training_x))
        correct = np.sum(y_predict == self.training_y)
        if correct == len(y_predict):
            self.wrong_list = None
        else:
            self.wrong_list = np.where(y_predict != self.training_y)[0]
        print(f"{correct} out of {len(y_predict)} training data correct, {len(y_predict) - correct} out of {len(y_predict)} training data wrong")
        print(f"correct rate = {correct / len(y_predict)}\nwrong rate = {(len(y_predict) - correct) / len(y_predict)}")

class OneVs(Enum):
    All = 0
    One = 1

class MultiSVM(object):

    def __init__(self, training_data, test_data=None, kernel=linear_kernel, C=None, multi_type=OneVs.All):
        # multi_type
        self.multi_type = multi_type
        self.kernel = kernel
        self.C = float(C) if C is not None else C
        self.training_x = training_data[:, :-1]
        self.training_y = training_data[:, -1]
        if test_data is not None:
            self.test_x = test_data[:, :-1]
            self.test_y = test_data[:, -1]
        # ?????????????????????
        self.svm_set = dict()
        self.classes = set(self.test_y)
        if multi_type == OneVs.All:
            self.train = self._one_vs_all
        elif multi_type == OneVs.One:
            self.train = self._one_vs_one

    def _one_vs_one(self):
        self.svm_set = dict()
        x = self.training_x
        self.all_pairs = list(combinations(self.classes, 2))

        for pair in self.all_pairs:
            training_label = self.training_y.copy()
            pos_mask = training_label == pair[0]
            neg_mask = training_label == pair[1]
            training_label[pos_mask] = 1
            training_label[neg_mask] = -1
            sub_mask = np.logical_or(pos_mask, neg_mask)
            sub_data = np.hstack((x[sub_mask], training_label[sub_mask].reshape(-1, 1)))
            svm = SVM(training_data=sub_data, kernel=self.kernel, C=self.C)
            svm.train()
            self.svm_set[pair] = svm

    def _one_vs_all(self):
        self.svm_set = []
        k = len(self.classes)
        # ????????????class????????????svm????????????, ??????one???rest
        for i in range(1, k + 1):
            training_label = self.training_y.copy()
            pos_mask = training_label == i
            training_label[pos_mask] = 1
            training_label[np.invert(pos_mask)] = -1
            svm = SVM(np.hstack((self.training_x, training_label.reshape(-1, 1))), kernel=self.kernel, C=self.C)
            svm.train()
            self.svm_set.append(svm)

    def test(self):
        y_predict = []
        if self.multi_type == OneVs.One:
            # one vs one: ????????????????????????, ?????????
            vote = np.zeros((len(self.test_y), len(self.classes)))
            for pair, svm in self.svm_set.items():
                pair = np.array(pair).astype(np.int) - 1
                y_predict = svm.project(self.test_x)
                first_mask = y_predict > 0
                for i, mask in enumerate(first_mask):
                    if mask:
                        vote[i][pair[0]] += 1
                    else:
                        vote[i][pair[1]] += 1
            y_predict = np.argmax(vote, axis=1) + 1
            # ????????????????????????????????????
            hits = y_predict == self.test_y
            misclassified = np.logical_not(hits)
            wrong_index = np.argwhere(misclassified).ravel()
            # ???????????????
            correct = np.sum(hits)
            print(f"{correct} out of {len(y_predict)} predictions correct, {len(y_predict) - correct} out of {len(y_predict)} predictions wrong")
            print(f"correct rate = {correct / len(y_predict)}\nwrong rate = {(len(y_predict) - correct) / len(y_predict)}")
            # ??????????????????
            true_labels = self.test_y[misclassified].astype(int)
            wrong_labels = y_predict[misclassified]
            return correct / len(y_predict), list(zip(wrong_index, true_labels, wrong_labels))

        elif self.multi_type == OneVs.All:
            # one vs all: ??????????????????????????????????????????
            for svm in self.svm_set:
                y_predict.append(svm.project(self.test_x))
            y_predict = np.argmax(y_predict, axis=0) + 1
            # ????????????????????????????????????
            hits = y_predict == self.test_y
            misclassified = np.logical_not(hits)
            wrong_index = np.argwhere(misclassified).ravel()
            # ???????????????
            correct = np.sum(hits)
            print(f"{correct} out of {len(y_predict)} predictions correct, {len(y_predict) - correct} out of {len(y_predict)} predictions wrong")
            print(f"correct rate = {correct / len(y_predict)}\nwrong rate = {(len(y_predict) - correct) / len(y_predict)}")
            # ??????????????????
            true_labels = self.test_y[misclassified].astype(int)
            wrong_labels = y_predict[misclassified]
            return correct / len(y_predict), list(zip(wrong_index, true_labels, wrong_labels))

if __name__ == "__main__":
    img_count = 0
    for i in range(1, 3):
        training_data = np.loadtxt(f"data5/training_{i}.txt")
        test_data = np.loadtxt(f"data5/test_{i}.txt")
        c = np.arange(0, 1.1, 0.1)
        out_count = 0
        plt.figure(img_count)
        for j in range(len(c)):
            print("-------------training_data{}, c = {:.2}---------------".format(i, c[j]))
            svm = SVM(training_data, test_data, C=c[j])
            svm.train()
            svm.training_error()
            svm.test()
            svm.plot_margin()
            # save images
            print('save')
            plt.savefig(f"????????????/pic/training_single_{i}_{out_count}.png", dpi=500)
            img_count += 1
            out_count += 1

    plt.close()
