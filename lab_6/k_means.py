import numpy as np
from numpy import linalg
from tqdm import trange
import cv2

class Kmeans(object):

    def __init__(self, source_image, k=16):
        self.training_x = source_image.copy().reshape(-1, 3)
        self.label = np.zeros(self.training_x.shape[0])
        self.mu = np.random.randint(low=0, high=255, size=(k, 3)).astype(np.float64)
        self.k = k

    def train(self):
        x = self.training_x
        epsilon = 1e-5
        loop_count = 0
        # 最多loop 100次
        for loop in trange(100):
            loop_count += 1
            pre_mu = self.mu.copy()
            # 计算每个sample对应的group
            for i, xi in enumerate(x):
                # 选取对于sample xi来说，最近的group作为label，距离度量使用L2-Norm ^ 2
                self.label[i] = np.argmin(linalg.norm(xi - self.mu, axis=1, keepdims=True) ** 2)
            # 更新每个group的 mu_j 对应的颜色
            for j in range(self.k):
                # 判断group j是否有数据点
                if (self.label == j).any():
                    self.mu[j] = np.sum(x[self.label == j], axis=0) / np.sum(self.label == j)

            if (linalg.norm(self.mu - pre_mu, axis=1, keepdims=True) < epsilon).all():
                break

        return loop_count

    def reassign(self, img):
        new_img = img.copy().astype(np.float64)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k = np.argmin(linalg.norm(new_img[i][j] - self.mu, axis=1, keepdims=True) ** 2)
                new_img[i][j] = self.mu[k]

        return new_img

if __name__ == "__main__":
    large_img = cv2.imread('data6/bird_large.tiff')
    small_img = cv2.imread('data6/bird_small.tiff')
    for k in [16, 32, 64, 128]:
        print(f'{k} bit colors')
        for i in range(10):
            kmeans = Kmeans(small_img, k=k)
            kmeans.train()
            new_img = kmeans.reassign(large_img)
            cv2.imwrite('data6/bird_large_after_{}_with_k_{}.tiff'.format(i, k), np.uint8(np.round(new_img)))
