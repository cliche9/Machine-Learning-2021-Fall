import numpy as np
import matplotlib.pyplot as plt

def normalize(data_in):
    data_min = np.min(data_in)
    data_max = np.max(data_in)
    data_out = (data_in - data_min) / (data_max - data_min)
    return data_out

class PCA(object):

    def __init__(self, ratio):
        self.n_ratio = ratio
        self.U = None
        self.sigma = None

    def fit_transform(self, x):
        u, s, vt = self._fit_svd(x)
        self.explained_variance_ = s ** 2 / (x.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        self.components_ = vt
        self.n_components = self.n_components_by_ratio()
        index = range(self.n_components)
        self.U = vt.T[:, index]
        x_pca = np.dot(x, self.U)
        return x_pca

    # 可视化特征脸
    def get_eigen_faces(self):
        faces = self.components_[:36, :]
        fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                                 subplot_kw={'xticks': [], 'yticks': []},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces[i].reshape(112, -1), cmap='bone')
        plt.savefig('data7/pic/top_36_eigen_faces.png')
        plt.close()

    def transform(self, x):
        mu = np.mean(x, axis=0)
        x -= mu
        x_pca = np.dot(x, self.U)
        return x_pca

    def _fit_eig(self, data_in, n_components):
        # step1: 将所有的数据点移到均值中心
        mu = np.mean(data_in, axis=0)
        data_in -= mu
        # step2: 计算协方差矩阵
        S = np.dot(data_in.T, data_in) / data_in.shape[0]
        # step3: 矩阵S的特征值分解
        vals, vecs = np.linalg.eig(S)
        # 选取前K大特征值对应的特征向量组成投影矩阵U
        index = range(n_components)
        U = np.real(vecs[:, index])
        # 将原数据投影到新空间返回新数据
        data_out = np.dot(data_in, U)
        return data_out, U

    def _fit_svd(self, data_in):
        # step1: 将所有的数据点移到均值中心
        mu = np.mean(data_in, axis=0)
        data_in -= mu
        """
        sample = mu.reshape(112, -1).astype(np.uint8)
        cv2.imshow('Average', sample)
        cv2.waitKey()
        """
        # step2: data_in的SVD分解, 直接得到feature的特征向量
        u, sigma, vt = np.linalg.svd(data_in, full_matrices=False)

        return u, sigma, vt

    def n_components_by_ratio(self):
        # 根据特征占比求投影维度
        s_sum = np.sum(self.explained_variance_)
        t = 0
        for k, val in enumerate(self.explained_variance_):
            if t > s_sum * self.n_ratio:
                return k + 1
            else:
                t += val

    def show_ratio(self):
        ratio_list = np.cumsum(self.explained_variance_ratio_)
        plt.figure()
        plt.plot(range(1, self.explained_variance_.shape[0] + 1), ratio_list * 100)
        plt.xlabel('Dimensions')
        plt.ylabel('Feature Ratio(%)')
        plt.ylim([0, 100])
        plt.grid()
        plt.savefig('data7/pic/Feature_Dimension.png')
        plt.close()
