import numpy as np
import argparse
from data_util import Dataset
from tqdm import tqdm, trange

def calculate_class_stat(train_data):
    mean = {}
    var = {}
    for i in range(train_data.shape[-1]):
        mean[i] = np.mean(train_data[:,:,i], axis=0)
        var[i] = np.std(train_data[:,:,i], axis=0) ** 2

    return mean, var

def gaussian(x, mean, var):
    return round((1 / np.sqrt(2 * np.pi * var)) * np.exp(((x - mean) ** 2) / (-2 * var)), 5)

def predict(feat, prob_class, classes, mean, var):
    probability = np.array([])
    prob = 1
    for j in range(classes):
        for i in range(feat.shape[0]):
            prob = prob * gaussian(feat[i], mean[j][i], var[j][i]) * prob_class

        probability = np.append(probability, prob)
        prob = 1

    return probability

def test(test_data, mean, var):
    total_samples = test_data.shape[0] * test_data.shape[-1]
    correct = 0
    prob_class = 1 / test_data.shape[-1]
    for i in trange(test_data.shape[-1], desc='testing'):
        for j in range(test_data.shape[0]):
            prediction = predict(test_data[j, :, i], prob_class, test_data.shape[-1], mean, var)
            if prediction.argmax() == i:
                # print(i)
                correct = correct + 1

    return correct*100 / total_samples

def bayes_classification(data):
    mean, var = calculate_class_stat(data.train_data)
    train_acc = test(data.train_data, mean, var)
    test_acc = test(data.test_data, mean, var)
    print(f"Training accuracy is: {train_acc}")
    print(f"Testing accuracy is: {test_acc}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Bayes")
    parser.add_argument('--data_name', type=str, default='data',
                        help='choose from data, pose and illum')
    args = parser.parse_args()
    data_name = args.data_name
    threshold = {'data': 0.02,
                 'pose': 0.025,
                 'illum': 0.1}
    data = Dataset()
    data.load_data(transform='PCA', threshold=threshold[data_name], data_name=data_name)
    data.train_val_test_split(data_name=data_name)
    bayes_classification(data)