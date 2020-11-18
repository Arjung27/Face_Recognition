import numpy as np
import sys
import argparse
from data_util import Dataset
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class Error(Exception):
    pass

class MoreNeighbours(Error):
    pass

def nearest_neighbour(train_data, test_data, k=1):
    # print(k)
    test_data = test_data[None, :, None]
    norm_l2 = np.linalg.norm(train_data - test_data, axis=1)
    flattened = np.ravel(norm_l2.T)
    index = np.argsort(flattened)
    index = index[:k]
    index = [int(i / train_data.shape[0]) for i in index]
    counts = np.bincount(index)
    # print(counts)

    # Breaking the tie
    max_count = np.max(counts)
    inds = np.where(counts == max_count)[0]
    np.random.seed(10)
    if len(inds) > 1:
        random = np.random.rand(len(inds))
        index = np.argmax(random)
        return inds[index]

    else:
        return np.argmax(counts)

def knn_classification(data, k=1):

    try:
        if k > data.train_data.shape[0]:
            raise MoreNeighbours
    except MoreNeighbours:
        print("More nearest neighbours needed than data in any class")
        sys.exit()
    correct = 0
    total_samples = data.test_data.shape[-1] * data.test_data.shape[0]
    for i in trange(data.test_data.shape[-1], desc='testing'):
        for j in range(data.test_data.shape[0]):

            distance = nearest_neighbour(data.train_data, data.test_data[j, :, i], k)

            if distance == i:
                correct = correct + 1

    acc = correct * 100 / total_samples
    print(f"Test accuracy: {acc}")

    return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--data_name', type=str, default='data',
                        help='choose from data, pose and illum')
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--transform', type=str, default='PCA', help='PCA or MDA')
    args = parser.parse_args()
    data_name = args.data_name
    # threshold = {'data': 0.3,
    #              'pose': 0.08,
    #              'illum': 0.1}
    # k = {'data': 1,
    #      'pose': 3,
    #      'illum': 1}
    # data = Dataset()
    # data.load_data(transform='PCA', threshold=threshold[data_name], data_name=data_name)
    # data.train_val_test_split(data_name=data_name)
    # knn_classification(data, k=k[data_name])

    test_acc_list = {}
    split = []
    if data_name =='data':
        indexes = np.arange(0.005, 0.3, 0.01)
    else:
        indexes = np.arange(0.005, 0.1, 0.01)
    for _, j in enumerate(indexes):
        if args.task_id == 1:

            threshold = {'data': j,
                         'pose': j,
                         'illum': j}

        elif args.task_id == 2:
            threshold = {'data': j}

        data = Dataset(task_id=args.task_id)
        data.load_data(transform=args.transform, threshold=threshold[data_name], data_name=data_name)
        data.train_val_test_split(data_name=data_name)

        for i in range(min(data.train_data.shape[0], 10)):
            k = {'data': i + 1,
                 'pose': i + 1,
                 'illum': i + 1}
            test_acc = knn_classification(data, k=k[data_name])
            if i + 1 in test_acc_list.keys():
                test_acc_list[i + 1].append(test_acc)
            else:
                test_acc_list[i + 1] = [test_acc]

    split = list(indexes)

    fig = plt.figure()
    for keys in test_acc_list.keys():
        plt.plot(split, test_acc_list[keys], label=f'k={keys}')
        plt.legend(bbox_to_anchor=(0.82, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel('Fraction of Principal Components Taken')
        plt.ylabel('Test Accuracy')

    plt.savefig(f'./Dataset/{data_name}/knn/test_acc_transform={args.transform}_taskid={args.task_id}.png')
    plt.close()