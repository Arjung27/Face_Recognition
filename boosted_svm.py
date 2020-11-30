import numpy as np
import sys
import argparse
from data_util import Dataset
from tqdm import tqdm, trange
from kernel_svm import svm
import matplotlib.pyplot as plt
from colorama import Fore
from kernel_svm import svm
from scipy.interpolate import interp1d

class Adaboost():

    def __init__(self, data, n_iters, kernel, max_epochs=10000, lr=0.00008,
                 error_threshold=0.5):
        self.data = data
        self.train_data = data.train_data
        self.test_data = data.test_data
        self.n_iters = n_iters
        self.classifiers = []
        self.kernel = kernel
        self.max_epochs = max_epochs
        self.lr = lr
        self.accs = []
        self.error_threshold = error_threshold

    def fit(self):
        self.X_train = np.reshape(self.train_data, [-1, self.train_data.shape[1]])
        labels = np.array([-1] * self.train_data.shape[0] + [1] * self.train_data.shape[0])
        labels = np.expand_dims(labels, 1)
        total_data = self.X_train.shape[0]
        w = np.full(total_data, (1 / total_data))

        for i in trange(self.n_iters, desc='boosting progress',
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):

            count = 0
            error = np.inf
            while error > self.error_threshold:
                obj = svm(self.data, self.kernel, max_epochs=self.max_epochs, lr=self.lr,
                                 margin=10000, seed=False, val=False, boost=True,
                                 kernel_param=2)
                obj.svm_classification()
                error = np.sum(w[(labels != np.sign(obj.prediction)).squeeze()])
                if error > self.error_threshold:
                    continue

                obj.alpha = 0.5 * np.log((1.0 - error) / (error + 1e-12))
                predictions = np.ones(np.shape(labels))
                wrong_pred = labels != obj.prediction
                predictions[wrong_pred] = -1
                w *= np.exp(-obj.alpha * labels.squeeze() * np.sign(obj.prediction).squeeze())
                w /= np.sum(w)
                self.classifiers.append(obj)
                self.predict_boost('test')
                self.accs.append(self.acc)

    def predict_boost(self, mode='test'):
        if mode == 'test':
            X_test = np.reshape(self.test_data, [-1, self.test_data.shape[1]])
            labels = np.array([-1] * self.test_data.shape[0] + [1] * self.test_data.shape[0])

        elif mode == 'val':
            X_test = np.reshape(self.val_data, [-1, self.val_data.shape[1]])
            labels = np.array([-1] * self.val_data.shape[0] + [1] * self.val_data.shape[0])

        else:
            X_test = np.reshape(self.train_data, [-1, self.train_data.shape[1]])
            labels = np.array([-1] * self.train_data.shape[0] + [1] * self.train_data.shape[0])

        # labels = np.expand_dims(labels, 1)
        total_data = X_test.shape[0]
        y_pred = np.zeros((total_data, 1))
        for cls in self.classifiers:
            cls_prediction = cls.predict(mode=mode)
            y_pred += cls.alpha * cls_prediction

        y_pred = np.sign(y_pred).flatten()
        acc = y_pred == labels
        self.acc = np.sum(acc) / total_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--data_name', type=str, default='data',
                        help='data')
    parser.add_argument('--iterations', type=int, default=15, help='Number of classifiers')
    parser.add_argument('--kernel', type=str, default='linear', help='rbf or poly')
    parser.add_argument('--max_epochs', type=int, default=10000, help='Number of classifiers')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate for SVM')
    parser.add_argument('--thresh', type=float, default=0.5, help='Upper bound for error')
    parser.add_argument('--transform', type=str, default='PCA', help='PCA or MDA')
    args = parser.parse_args()

    if args.kernel == 'linear':
        args.thresh = 0.5

    if args.lr is None:
        lr = {'PCA': 2.56e-5,
              'MDA': 4.09e-04}
        args.lr = lr[args.transform]

    data_name = 'data'
    threshold = {'data': 0.02}
    data = Dataset(task_id=2)
    data.load_data(transform=args.transform, threshold=threshold[data_name], data_name='data')
    data.train_val_test_split(data_name=data_name, test_ratio=0.8, seed=False)
    # for iter in args.iterations:
    boost = Adaboost(data, args.iterations, kernel=args.kernel, max_epochs=args.max_epochs,
                     error_threshold=args.thresh, lr=args.lr)
    boost.fit()

    # fig = plt.figure()
    # x_ticks = list(np.arange(1, len(boost.accs) + 1))
    # x_ticks1 = list(np.arange(1, len(boost.accs) + 1, max(1, int((len(boost.accs) + 1)/ 10))))
    # y_ticks1 = list(np.arange(0.2, 1.0, 0.1))
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.xticks(x_ticks1)
    # plt.yticks(y_ticks1)
    # plt.xlim(np.min(x_ticks1), np.max(x_ticks1))
    # plt.ylim(np.min(y_ticks1), np.max(y_ticks1))
    # plt.plot(x_ticks, boost.accs)
    # plt.plot(x_ticks, boost.accs, 'g*')
    # plt.savefig(f'boosted_svm_transform={args.transform}_epochs={boost.max_epochs}_'
    #             f'iterations={args.iterations}_lr={boost.lr}_kernel={args.kernel}_thresh={args.thresh}.png')
