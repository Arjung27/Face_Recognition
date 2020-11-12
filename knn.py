import numpy as np
import sys
import argparse
from data_util import Dataset
from tqdm import tqdm, trange

class Error(Exception):
    pass

class MoreNeighbours(Error):
    pass

def nearest_neighbour(train_data, test_data, k=1):
    # print(k)
    test_data = test_data[None, :, None]
    norm_l2 = np.linalg.norm(train_data - test_data, axis=1)
    flattened = np.ravel(norm_l2.T)
    index = np.argpartition(flattened, k)
    index = index[:k]
    # print(f'index before: {index}')
    index = [int(i / train_data.shape[0]) for i in index]
    counts = np.bincount(index)
    # print(f'index : {index}')
    # print(f'count : {counts}')
    # print(np.argmax(counts))
    # exit(-1)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--data_name', type=str, default='data',
                        help='choose from data, pose and illum')
    args = parser.parse_args()
    data_name = args.data_name
    threshold = {'data': 0.3,
                 'pose': 0.08,
                 'illum': 0.1}
    k = {'data': 1,
         'pose': 3,
         'illum': 1}
    data = Dataset()
    data.load_data(transform='PCA', threshold=threshold[data_name], data_name=data_name)
    data.train_val_test_split(data_name=data_name)
    knn_classification(data, k=k[data_name])
