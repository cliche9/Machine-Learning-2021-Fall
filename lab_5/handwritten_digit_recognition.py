import numpy as np
import matplotlib.pyplot as plt
from svm import SVM

def str2img(file_path):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    data_x = []
    data_y = []
    for line in data:
        str_list = line.strip().split(' ')
        data_y.append(int(str_list[0]))
        t = np.zeros(784)
        for p in str_list[1:]:
            a, b = map(int, p.split(':'))
            t[a] = b

        t = t.reshape(28, 28)
        data_x.append(t)

    return data_x, data_y

def img2vec(source_images):
    n = source_images.shape[0]
    img_normalized = np.round(source_images / 255, 3)
    img_vec = img_normalized.reshape(n, -1)

    return img_vec

def down_sampling(source_images, pooling_size=4, type='avg'):
    if (source_images.shape[1] % pooling_size != 0):
        pass
    n, w, h = source_images.shape
    new_images = np.empty((n, int(w / pooling_size), int(h / pooling_size)))
    if (type == 'avg'):
        pooling_function = np.mean
    elif (type == 'max'):
        pooling_function = np.max
    for k in range(n):
        for i in range(new_images.shape[1]):
            for j in range(new_images.shape[2]):
                pooling_region = source_images[k][i*pooling_size:(i+1)*pooling_size, j*pooling_size:(j+1)*pooling_size]
                new_images[k][i][j] = pooling_function(pooling_region)

    return new_images

if __name__ == "__main__":
    # 图像预处理为向量
    print("----------Loading training data and test data...----------")
    img_training_x, img_training_y = str2img('data5/train-01-images.svm')
    img_test_x, img_test_y = str2img('data5/test-01-images.svm')
    print("----------Done----------")
    print("----------Preprocessing images with down sampling...----------")
    vec_training_x = img2vec(down_sampling(np.array(img_training_x)))
    vec_test_x =img2vec(down_sampling(np.array(img_test_x)))

    training_data = np.column_stack((vec_training_x, img_training_y))
    test_data = np.column_stack((vec_test_x, img_test_y))
    print("----------Done----------")
    # 建立SVM
    Cs = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in Cs:
        print(f"----------Construct SVM model with C = {c}...----------")
        svm = SVM(training_data, test_data, C=c)
        print("----------Done----------")
        print("----------Start training SVM...----------")
        svm.train()
        print("----------Training SVM done----------")
        print("----------Start retesting training data...----------")
        svm.training_error()
        # 最多保存3张误分类图
        if svm.wrong_list is not None:
            n = len(svm.wrong_list) if len(svm.wrong_list) < 3 else 3
            for i in range(n):
                if c is None:
                    plt.imsave('实验报告/pic/c_None/wrong_{}.png'.format(i), img_training_x[svm.wrong_list[i]], cmap='gray')
                else:
                    plt.imsave('实验报告/pic/c_{}/wrong_{}.png'.format(c, i), img_training_x[svm.wrong_list[i]], cmap='gray')
        print("----------Start testing SVM...----------")
        svm.test()

