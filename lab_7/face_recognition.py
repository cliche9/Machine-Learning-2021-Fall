import os
import random
import sys
sys.path.append('..')
import numpy as np
import cv2
from pca import PCA
from lab_5 import svm
import matplotlib.pyplot as plt

def load_data(get_avg_faces=False):
    path = 'data7/orl_faces/'
    dirs = list(filter(lambda f: f != 'README', os.listdir(path)))
    training_imgs = []
    test_imgs = []
    training_labels = []
    test_labels = []
    all_imgs = []

    index = list(range(10))
    for i, dir in enumerate(dirs):
        all_imgs.append([])
        # 读取全部图片, 将其展开成向量
        for j in range(1, 11):
            img = np.ravel(cv2.imread(path + dir + f'/{j}.pgm', cv2.IMREAD_GRAYSCALE))
            all_imgs[i].append(img)
        all_imgs[i] = np.array(all_imgs[i])
        nums = random.choice(range(5, 8))
        random.shuffle(index)
        # 获取每一类训练数据的平均脸
        if get_avg_faces:
            faces = np.vstack(all_imgs[i][index[:nums]])
            avg_face = np.mean(faces, axis=0).reshape(112, -1)
            label = dir
            fig = plt.figure(i)
            plt.imshow(avg_face, cmap='gray')
            plt.title(f'Average face of {dir}')
            plt.savefig(f'data7/pic/avg_face_of_{dir}.png')

        training_imgs.append(all_imgs[i][index[:nums]])
        test_imgs.append(all_imgs[i][index[nums:]])
        for j in range(nums):
            training_labels.append(int(dir[1:]))
        for j in range(nums, 10):
            test_labels.append(int(dir[1:]))

    training_labels = np.array(training_labels).reshape(-1, 1)
    test_labels = np.array(test_labels).reshape(-1, 1)

    return np.vstack(training_imgs).astype(np.float64), training_labels, np.vstack(test_imgs).astype(np.float64), test_labels

# 获取误分类情况并保存成图片
def get_misclassified_faces(training_imgs, training_labels, test_imgs, test_labels):
    # PCA降维
    pca = PCA(0.9)
    training_data = np.hstack((pca.fit_transform(training_imgs.copy()), training_labels))
    test_data = np.hstack((pca.transform(test_imgs.copy()), test_labels))
    # 获取特征所占比例
    pca.show_ratio()
    # 获取36个特征脸
    pca.get_eigen_faces()
    # 建立多分类SVM
    multi_svm = svm.MultiSVM(training_data, test_data, kernel=svm.linear_kernel, C=0, multi_type=svm.OneVs.One)
    # 训练
    multi_svm.train()
    # 预测
    correct, wrong_list = multi_svm.test()
    # 将误分类的图像保存下来
    for i, img_info in enumerate(wrong_list):
        index, true_label, wrong_label = img_info
        fig = plt.figure(i)
        plt.imshow(test_imgs[index].reshape(112, -1), cmap='gray')
        plt.title(f's{true_label} is misclassified as s{wrong_label}')
        plt.savefig(f'data7/pic/misclassified_{i}.png')
        plt.close()

# 改变K值, 观察准确率的变化
def acc_with_k_range(training_imgs, training_labels, test_imgs, test_labels):
    # PCA降维
    ratio_list = np.arange(0.1, 1.0, 0.1)
    correct_list = []
    for i, type in enumerate(svm.OneVs):
        correct_list.append([])
        for ratio in np.arange(0.1, 1.0, 0.1):
            pca = PCA(ratio)
            training_data = np.hstack((pca.fit_transform(training_imgs), training_labels))
            # pca.show_ratio()
            test_data = np.hstack((pca.transform(test_imgs), test_labels))
            # 建立多分类SVM
            multi_svm = svm.MultiSVM(training_data, test_data, kernel=svm.linear_kernel, C=0, multi_type=type)
            # 训练
            multi_svm.train()
            # 预测
            correct, _ = multi_svm.test()
            correct_list[i].append(correct)

    plt.figure()
    plt.plot(ratio_list, correct_list[0], label='OVR')
    plt.plot(ratio_list, correct_list[1], label='OVO')
    plt.legend()
    plt.xlabel('Feature_Ratio')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('data7/pic/OVO_OVR.png')
    plt.close()

if __name__ == "__main__":
    training_imgs, training_labels, test_imgs, test_labels = load_data(get_avg_faces=False)
    get_misclassified_faces(training_imgs, training_labels, test_imgs, test_labels)
    # acc_with_k_range(training_imgs, training_labels, test_imgs, test_labels)