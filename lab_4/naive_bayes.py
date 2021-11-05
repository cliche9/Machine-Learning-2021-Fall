import numpy as np
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self, training_data, test_data):
        self.training_x = training_data[:, :-1]
        self.training_y = training_data[:, -1]
        self.test_x = test_data[:, :-1]
        self.test_y = test_data[:, -1]
        self.py = np.zeros(5)
        self.pjxy = np.zeros((8, 5, 5))

    def get_py_with_ls(self, count_y, batch_size, number_of_value_y):
        self.py = np.zeros(5)
        for i in range(5):
            self.py[i] = (count_y[i] + 1) / (batch_size + number_of_value_y)

    def get_pjxy_with_ls(self, count_jxy, count_y, number_of_value_x):
        self.pjxy = np.zeros((8, 5, 5))
        for j in range(8):
            for x in range(5):
                for y in range(5):
                    self.pjxy[j][x][y] += (count_jxy[j][x][y] + 1) / (count_y[y] + number_of_value_x[x])


    def max_likelihood(self, x):
        pred_y = self.py.copy()
        for y in range(5):
            for j in range(8):
                pred_y[y] *= self.pjxy[j][x[j]][y]

        return np.argmax(pred_y)

    def train(self, training_data):
        self.training_x = training_data[:, :-1]
        self.training_y = training_data[:, -1]

        # count_x[j][x]: 第j个feature, xj = x的个数
        count_x = np.zeros((8, 5))
        # count_y[y]: y_label = y的个数
        count_y = np.zeros(5)
        # count_jxy[j][x][y]: 第j个feature，xj = x and y = y的个数
        count_jxy = np.zeros((8, 5, 5))

        # 遍历所有training_data, 记录每一行中count_x和count_y
        for data_x, data_y in zip(self.training_x, self.training_y):
            count_y[data_y] += 1
            for j in range(8):
                count_x[j][data_x[j]] += 1
                count_jxy[j][data_x[j]][data_y] += 1

        number_of_value_x = np.array([np.sum(count_x[j] > 0) for j in range(8)])
        number_of_value_y = np.sum(count_y > 0)

        self.get_py_with_ls(count_y, training_data.shape[0], number_of_value_y)
        self.get_pjxy_with_ls(count_jxy, count_y, number_of_value_x)

    def predict(self, batch_size):
        right_count = 0
        m = self.test_x.shape[0]
        for data_x, data_y in zip(self.test_x, self.test_y):
            pred_y = self.max_likelihood(data_x)
            if data_y == pred_y:
                right_count += 1

        print(f'accuracy of batch size = {batch_size}: {right_count / m}')

        return right_count / m

if __name__ == "__main__":
    # load data & initial model
    training_data = np.loadtxt("data4/training_data.txt", dtype=int)
    test_data = np.loadtxt("data4/test_data.txt", dtype=int)

    nb = NaiveBayes(training_data, test_data)
    batch_size = np.arange(100, 1000, 100)
    batch_size = np.concatenate((batch_size, np.arange(1000, 11000, 1000)))

    test_acc_list = []
    for size in batch_size:
        print(size)
        np.random.shuffle(training_data)
        nb.train(training_data[:size, :])
        test_acc_list.append(nb.predict(size))

    plt.figure(1)
    plt.grid()
    plt.ylim([0.5, 1])
    plt.xlabel('Batch size')
    plt.ylabel('Accuracy')
    plt.plot(batch_size, test_acc_list)
    plt.show()


