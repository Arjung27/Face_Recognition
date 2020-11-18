import numpy as np
import argparse
from data_util import Dataset
from tqdm import tqdm, trange
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

def calculate_class_stat(train_data):
    mean = {}
    cov = {}

    for i in range(train_data.shape[-1]):
        mean[i] = np.mean(train_data[:,:,i], axis=0)
        mean_ = np.expand_dims(mean[i], -1)
        cov_ = 0
        for j in range(train_data[:,:,i].shape[0]):
            x = np.expand_dims(train_data[j,:,i], -1)
            cov_ += np.dot(x - mean_, (x - mean_).T)

        cov[i] = (1/(j+1)) * cov_
        X = np.random.multivariate_normal(mean=mean[i],
                                          cov = cov[i],
                                          size = 50)
        covLW= LedoitWolf().fit(X)
        cov[i] = covLW.covariance_

    return mean, cov

def gaussian(x, mean, cov):
    k = cov.shape[0]
    # print(np.linalg.det(cov))
    det = np.linalg.det(cov)
    const = (2 * np.pi) ** (-k/2) * det ** (-1/2)
    x = np.expand_dims(x, -1)
    mean = np.expand_dims(mean, -1)
    diff = x - mean
    power = (-1/2) * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)
    return const * np.exp(power)

def predict(feat, prob_class, classes, mean, cov):
    probability = np.array([])
    prob = 1
    for j in range(classes):
        prob = prob * gaussian(feat, mean[j], cov[j]) * prob_class
        probability = np.append(probability, prob)
        prob = 1

    return probability

def test(test_data, mean, cov):
    total_samples = test_data.shape[0] * test_data.shape[-1]
    correct = 0
    prob_class = 1 / test_data.shape[-1]
    for i in trange(test_data.shape[-1], desc='testing'):
        for j in range(test_data.shape[0]):
            prediction = predict(test_data[j, :, i], prob_class, test_data.shape[-1], mean, cov)
            if prediction.argmax() == i:
                # print(i)
                correct = correct + 1

    return correct*100 / total_samples

def bayes_classification(data):
    mean, cov = calculate_class_stat(data.train_data)
    train_acc = test(data.train_data, mean, cov)
    test_acc = test(data.test_data, mean, cov)
    print(f"Training accuracy is: {train_acc}")
    print(f"Testing accuracy is: {test_acc}")

    return train_acc, test_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--data_name', type=str, default='data',
                        help='choose from data, pose and illum')
    parser.add_argument('--task_id', type=int, default=1)
    args = parser.parse_args()
    data_name = args.data_name
    test_acc_list = []
    split = []
    fig = plt.figure()
    for j in np.arange(0.02, 0.15, 0.01):
        if args.task_id == 1:
            # threshold = {'data': 0.03,
            #              'pose': 0.02,
            #              'illum': 0.02}

            threshold = {'data': j,
                         'pose': j,
                         'illum': j}

        elif args.task_id == 2:
            threshold = {'data': j}

        data = Dataset(task_id=args.task_id)
        data.load_data(transform='PCA', threshold=threshold[data_name], data_name=data_name)
        data.train_val_test_split(data_name=data_name)
        _, test_acc = bayes_classification(data)
        test_acc_list.append(test_acc)
        split.append(j)

        # plt.plot(split, test_acc_list)
        # plt.xlabel('Fraction of Principal Components Taken')
        # plt.ylabel('Test Accuracy')
        # plt.savefig(f'./Dataset/{data_name}/test_acc_taskid={args.task_id}.png')

    plt.close()
