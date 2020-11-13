import numpy as np
import sys
import argparse
from data_util import Dataset
from tqdm import tqdm, trange
from kernel_svm import svm
import matplotlib.pyplot as plt
from colorama import Fore
from scipy.interpolate import interp1d

class Adaboost(svm):

    def __init__(self, data, n_iters, kernel, max_epochs=10000, lr=0.00008, sig_step=-1,
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
        self.sig_step = sig_step
        self.error_threshold = error_threshold
        self.epochs = np.arange(50, (self.n_iters + 1)*50, 50)

    def fit(self):
        self.X_train = np.reshape(self.train_data, [-1, self.train_data.shape[1]])
        labels = np.array([-1] * self.train_data.shape[0] + [1] * self.train_data.shape[0])
        labels = np.expand_dims(labels, 1)
        total_data = self.X_train.shape[0]
        w = np.full(total_data, (1 / total_data))
        val_acc = 0
        prev_val_acc = 0.1
        for i in trange(self.n_iters, desc='boosting progress',
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):

            count = 0
            if self.kernel == 'rbf':
                kernel_param = self.train_data.shape[1]
            else:
                kernel_param = 2
            error = np.inf
            self.max_epochs = self.epochs[i]
            while error > self.error_threshold:
                super().__init__(self.data, self.kernel, max_epochs=self.max_epochs, lr=self.lr,
                                 margin=10000, seed=False, val=False, boost=True,
                                 kernel_param=kernel_param)
                self.svm_classification()
                error = np.sum(w[(labels != self.prediction).squeeze()])
                if error > self.error_threshold:
                    kernel_param += self.sig_step
                    kernel_param = max(1.5, kernel_param)
                    continue
                self.alpha = 0.5 * np.log((1.0 - error) / (error + 1e-12))
                predictions = np.ones(np.shape(labels))
                wrong_pred = labels != self.prediction
                predictions[wrong_pred] = -1
                prev_w = w.copy()
                w *= np.exp(-self.alpha * labels.squeeze() * predictions.squeeze())
                w /= np.sum(w)
                self.classifiers.append(self)
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
        # if mode == 'test':
        print(f"Accuracy on {mode} data: {self.acc * 100}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--iterations', type=int, default=6, help='Number of classifiers')
    parser.add_argument('--kernel', type=str, default='rbf', help='rbf or poly')
    parser.add_argument('--max_epochs', type=int, default=10000, help='Number of classifiers')
    parser.add_argument('--lr', type=float, default=0.00008, help='Learning rate for SVM')
    parser.add_argument('--thresh', type=float, default=0.5, help='Upper bound for error')
    parser.add_argument('--sig_step', type=float, default=1, help='Decay rate for sigma')
    args = parser.parse_args()

    if args.kernel == 'linear':
        args.thresh = 0.5

    data_name = 'data'
    threshold = {'data': 0.02}
    data = Dataset(task_id=2)
    data.load_data(transform='PCA', threshold=threshold[data_name], data_name='data')
    data.train_val_test_split(data_name=data_name, test_ratio=0.8, seed=False)
    # for iter in args.iterations:
    boost = Adaboost(data, args.iterations, kernel=args.kernel, max_epochs=args.max_epochs,
                     error_threshold=args.thresh, sig_step=args.sig_step)
    boost.fit()

    fig = plt.figure()
    x_ticks = list(np.arange(1, len(boost.accs) + 1))
    x_ticks1 = list(np.arange(1, len(boost.accs) + 1, max(1, int((len(boost.accs) + 1)/ 10))))
    y_ticks1 = list(np.arange(0.2, 1.0, 0.1))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(f'epochs={boost.max_epochs}_iters={args.iterations}_lr={boost.lr}'
              f'_kernel={args.kernel}')
    plt.xticks(x_ticks1)
    plt.yticks(y_ticks1)
    plt.xlim(np.min(x_ticks1), np.max(x_ticks1))
    plt.ylim(np.min(y_ticks1), np.max(y_ticks1))
    plt.plot(x_ticks, boost.accs)
    plt.plot(x_ticks, boost.accs, 'g*')
    plt.savefig(f'boosted_svm_epochs={boost.max_epochs}_iterations={args.iterations}_lr={boost.lr}'
                f'_kernel={args.kernel}_thresh={args.thresh}_sigstep={args.sig_step}.png')
