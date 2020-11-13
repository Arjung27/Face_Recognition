import numpy as np
import sys
import argparse
from data_util import Dataset
from tqdm import tqdm, trange

class svm:

    def __init__(self, data, kernel_key='rbf', max_epochs=500, lr=0.01, margin=10000,
                 eps=1e-05, seed=True, val=True, alpha=1, boost=False, **kwargs):
        self.data = data
        self.kernel_key = kernel_key
        self.train_data = data.train_data
        self.test_data = data.test_data
        self.val = val
        if val:
            self.val_data = data.val_data
        self.max_epochs = max_epochs
        self.lr = lr
        self.margin = margin
        self.kernel_map = {'rbf': 'rbf_kernel', 'poly': 'poly_kernel', 'linear': 'linear_kernel'}
        self.param = kwargs['kernel_param']
        self.eps = eps
        self.seed = seed
        self.alpha = alpha
        self.boost = boost

    def rbf_kernel(self, sigma=0.5):
        sigma = 1 / (sigma ** 2)
        kernel = lambda X, y: np.exp(sigma * np.square(X[:, None] - y).sum(axis=2))
        return kernel

    def poly_kernel(self, power=2):
        kernel = lambda X, y: ((np.matmul(X, y.T)) + 1) ** power
        return kernel

    def linear_kernel(self, power=1):
        kernel = lambda X, y: np.matmul(X, y.T)
        return kernel

    def calculate_gradient(self, x, y):
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
        distance = 1 - (y * np.dot(x, self.weights))
        distance[distance < 0] = 0
        dw = np.zeros(self.weights.shape[0])
        inds = np.where(distance > 0)[0]
        if len(inds) == 0:
            dw += self.weights
        else:
            dw += self.weights - (self.margin * y * x).squeeze()

        gradient = dw / len(self.weights)
        return gradient

    def gradient_descent(self, X, y):
        if self.seed:
            np.random.seed(10)
            self.weights = np.zeros(X.shape[1])
        else:
            self.weights = np.random.rand(X.shape[1])

        self.weights = np.zeros(X.shape[1])

        if not self.boost:
            for epoch in trange(self.max_epochs, desc='training svm'):
                shuffler = np.random.permutation(np.arange(y.shape[0]))
                X_shuffled = X[shuffler]
                y_shuffled = y[shuffler]
                weights = self.weights
                for batch_idx, data in enumerate(X_shuffled):
                    gradient = self.calculate_gradient(data, y_shuffled[batch_idx])
                    self.weights = self.weights - self.lr * gradient

                    if np.linalg.norm(gradient) <= self.eps:
                        return

                if ((epoch + 1) % 2000 == 0 or (epoch + 1) % 5000 == 0) and (self.val):
                    self.predict('train')
                    self.predict('val')
                    self.predict('test')
                    if (epoch + 1) % 5000 == 0:
                        self.lr = self.lr / 2
                        print(f'New lr ---> {self.lr}')
        else:
            for epoch in range(self.max_epochs):
                shuffler = np.random.permutation(np.arange(y.shape[0]))
                X_shuffled = X[shuffler]
                y_shuffled = y[shuffler]
                weights = self.weights
                for batch_idx, data in enumerate(X_shuffled):
                    gradient = self.calculate_gradient(data, y_shuffled[batch_idx])
                    self.weights = self.weights - self.lr * gradient

                    if np.linalg.norm(gradient) <= self.eps:
                        return

                if ((epoch + 1) % 2000 == 0 or (epoch + 1) % 5000 == 0) and (self.val):
                    self.predict('train')
                    self.predict('val')
                    self.predict('test')
                    if (epoch + 1) % 5000 == 0:
                        self.lr = self.lr / 2
                        print(f'New lr ---> {self.lr}')


    def svm_classification(self, **kwargs):
        self.X_train = np.reshape(self.train_data, [-1, self.train_data.shape[1]])
        labels = np.array([-1]*self.train_data.shape[0] + [1]*self.train_data.shape[0])
        labels = np.expand_dims(labels, 1)

        if self.kernel_key not in self.kernel_map.keys():
            raise ValueError('Give the correct kernel name. Choose from rbf or poly.')
            sys.exit()
        else:
            kernel = eval('self.' + self.kernel_map[self.kernel_key])(self.param)

        self.K = kernel(self.X_train, self.X_train)
        # Adding 1 for bias term
        self.K = np.insert(self.K, self.K.shape[1], 1, axis=1)
        self.gradient_descent(self.K, labels)
        trained_weights = np.expand_dims(self.weights, -1)
        self.prediction = np.sign(np.dot(self.K, trained_weights))

        if (self.val):
            self.predict('train')
            self.predict('val')

    def predict(self, mode='test'):
        if mode == 'test':
            X_test = np.reshape(self.test_data, [-1, self.test_data.shape[1]])
            labels = np.array([-1] * self.test_data.shape[0] + [1] * self.test_data.shape[0])
            labels = np.expand_dims(labels, 1)
        elif mode == 'val':
            X_test = np.reshape(self.val_data, [-1, self.val_data.shape[1]])
            labels = np.array([-1] * self.val_data.shape[0] + [1] * self.val_data.shape[0])
            labels = np.expand_dims(labels, 1)
        else:
            X_test = np.reshape(self.train_data, [-1, self.train_data.shape[1]])
            labels = np.array([-1] * self.train_data.shape[0] + [1] * self.train_data.shape[0])
            labels = np.expand_dims(labels, 1)

        kernel = eval('self.' + self.kernel_map[self.kernel_key])(self.param)
        self.K1 = kernel(X_test, self.X_train)
        self.K1 = np.insert(self.K1, self.K1.shape[1], 1, axis=1)
        trained_weights = np.expand_dims(self.weights, -1)

        if not self.boost:
            prediction = np.sign(np.dot(self.K1, trained_weights))
            prediction = prediction == labels
            acc = np.sum(prediction) * 100 / self.K1.shape[0]
            print(f"Accuracy on {mode} data: {acc}")
            return

        else:
            prediction = np.dot(self.K1, trained_weights)
            return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--kernel', type=str, default='rbf', help='rbf or poly')
    args = parser.parse_args()

    data_name = 'data'
    threshold = {'data': 0.02}
    data = Dataset(task_id=2)
    data.load_data(transform='PCA', threshold=threshold[data_name], data_name='data')
    data.train_val_test_split(data_name=data_name, test_ratio=0.8, cross_ratio=0.8)

    if args.kernel == 'rbf':
        param = data.train_data.shape[1]
        # param = 1
    else:
        param = 5

    ## RBF
    classifier = svm(data, args.kernel, max_epochs=10000, lr=0.00008, margin=10000, kernel_param=param)

    ## Poly
    # classifier = svm(data, args.kernel, max_epochs=10000, lr=0.00005, margin=10000, kernel_param=param)
    classifier.svm_classification()
    classifier.predict()
