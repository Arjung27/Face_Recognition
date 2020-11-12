import numpy as np
import sys
import argparse
from data_util import Dataset
from tqdm import tqdm, trange
from kernel_svm import svm
import matplotlib.pyplot as plt

class Adaboost(svm):

    def __init__(self, data, n_iters, kernel):
        self.data = data
        self.train_data = data.train_data
        self.test_data = data.test_data
        self.n_iters = n_iters
        self.classifiers = []
        self.kernel = kernel

    def fit(self):
        self.X_train = np.reshape(self.train_data, [-1, self.train_data.shape[1]])
        labels = np.array([-1] * self.train_data.shape[0] + [1] * self.train_data.shape[0])
        labels = np.expand_dims(labels, 1)
        total_data = self.X_train.shape[0]

        w = np.full(total_data, (1 / total_data))

        for i in range(self.n_iters):
            super().__init__(self.data, self.kernel, max_epochs=100, lr=0.00008, margin=10000,
                                 seed=False, val=False, kernel_param=self.train_data.shape[1])
            self.svm_classification()
            # print((labels != self.prediction).shape)
            error = np.sum(w[(labels != self.prediction).squeeze()])
            self.alpha = 0.5 * np.log((1.0 - error) / (error + 1e-12))
            predictions = np.ones(np.shape(labels))
            wrong_pred = labels != self.prediction
            predictions[wrong_pred] = -1
            w *= np.exp(-self.alpha * labels.squeeze() * predictions.squeeze())
            w /= np.sum(w)
            self.classifiers.append(self)

        self.predict_boost()

    def predict_boost(self):
        self.X_test = np.reshape(self.test_data, [-1, self.test_data.shape[1]])
        labels = np.array([-1] * self.test_data.shape[0] + [1] * self.test_data.shape[0])
        # labels = np.expand_dims(labels, 1)
        total_data = self.X_test.shape[0]
        y_pred = np.zeros((total_data, 1))

        for cls in self.classifiers:
            cls_prediction = cls.predict(boost=True)
            y_pred += cls.alpha * cls_prediction

        y_pred = np.sign(y_pred).flatten()
        acc = y_pred == labels
        self.acc = np.sum(acc) / total_data
        print(f"Accuracy on test data: {acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--iterations', type=int, default=5, help='Number of classifiers')
    parser.add_argument('--kernel', type=str, default='rbf', help='rbf or poly')
    args = parser.parse_args()

    data_name = 'data'
    threshold = {'data': 0.02}
    data = Dataset(task_id=2)
    data.load_data(transform='PCA', threshold=threshold[data_name], data_name='data')
    data.train_val_test_split(data_name=data_name, test_ratio=0.8, seed=False)
    boost = Adaboost(data, args.iterations, kernel=args.kernel)
    boost.fit()