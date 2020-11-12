import numpy as np
from scipy.io import loadmat
import os
import sys
from tqdm import trange

class Error(Exception):
    pass

class ZeroData(Error):
    pass

class Dataset:

    def __init__(self, data_path='Data', task_id=1, data_sub=200, pose_sub=68, illum_sub=68):
        self.data_path = data_path
        self.task_id = task_id
        self.data_sub = data_sub
        self.pose_sub = pose_sub
        self.illum_sub = illum_sub
        self.data_img_sub = 3
        self.pose_img_sub = 13
        self.illum_img_sub = 21
        self.processed_dataset = {}

    def load_data(self, transform='PCA', **kwargs):

        if self.task_id == 1:
            self.data = loadmat(os.path.join(self.data_path, 'data.mat'))['face']
            self.illumination = loadmat(os.path.join(self.data_path, 'illumination.mat'))['illum']
            self.pose = loadmat(os.path.join(self.data_path, 'pose.mat'))['pose']

            self.illum_data = {}
            self.pose_data = {}

            for i in range(self.pose_sub):
                self.pose_data[i] = \
                    self.pose[:, :, 0:self.pose_img_sub, i]

            for i in range(self.illum_sub):
                for j in range(self.illum_img_sub):
                    if i not in self.illum_data.keys():
                        self.illum_data[i] = \
                            np.reshape(self.illumination[:,  j, i], (48, 40))
                    else:
                        self.illum_data[i] = np.dstack((self.illum_data[i],
                                                       np.reshape(self.illumination[:,  j, i], (48, 40))))

            self.std_data = {}

            for i in range(self.data_sub):
                self.std_data[i] = \
                    self.data[:, :, 3 * i: 3 * (i + 1)]

        elif self.task_id == 2:
            # Since the second tak is a binary problem where each class has 200 images
            self.data_img_sub = 200
            self.data_sub = 2
            self.data = loadmat(os.path.join(self.data_path, 'data.mat'))['face']
            self.std_data = {}

            for i in range(self.data_sub):
                for j in range(self.data_img_sub):

                    if i == 0:
                        if i not in self.std_data.keys():
                            self.std_data[i] = self.data[:, :, 3*j]
                        else:
                            self.std_data[i] = np.dstack((self.std_data[i], self.data[:, :, 3*j]))
                    elif i == 1:
                        if i not in self.std_data.keys():
                            self.std_data[i] = self.data[:, :, 3*j + 1]
                        else:
                            self.std_data[i] = np.dstack((self.std_data[i], self.data[:, :, 3*j + 1]))

        self.transform_data(transform=transform, **kwargs)

    def make_data_dict(self):

        if self.task_id == 1:
            self.data_dict = {'data': [self.std_data, self.data_sub, self.data_img_sub],
                              'pose': [self.pose_data, self.pose_sub, self.pose_img_sub],
                              'illum': [self.illum_data, self.illum_sub, self.illum_img_sub]}
        elif self.task_id == 2:
            self.data_dict = {'data': [self.std_data, self.data_sub, self.data_img_sub]}

    def normalize_image(self, image):

        mean = np.mean(image)
        std = np.std(image)

        return (image.flatten() - mean) / std**2

    def transform_data(self, transform='PCA', **kwargs):

        self.make_data_dict()
        if transform.upper() == 'PCA':
            threshold = kwargs['threshold']
            data_name = kwargs['data_name']
            keys = data_name
            normalize_data = [self.normalize_image(self.data_dict[keys][0][i][:,:,j])
                              for i in range(self.data_dict[keys][1])
                              for j in range(self.data_dict[keys][2])]
            normalize_data = np.array(normalize_data, dtype=np.float64)
            cov_data = np.matmul(normalize_data.T, normalize_data)
            eig_value, eig_vector = np.linalg.eig(cov_data)

            # Sorting eigen value and corresponding eig vector in descending order
            index = np.argsort(eig_value)
            index = index[::-1]
            eig_value_sort = eig_value[index]
            eig_vector_sort = eig_vector[:, index]

            # Taking only real values
            eig_value_sort = eig_value_sort.real
            eig_vector_sort = eig_vector_sort.real

            principle_components = np.matmul(normalize_data, eig_vector_sort)
            # variance = np.std(principle_components, axis=0)
            # index = np.argsort(variance)

            try:
                col_threshold = int(principle_components.shape[-1] * threshold)
                if col_threshold <= 0:
                    raise ZeroData

            except ZeroData:
                print("Choose a higher threshold. Current value gives zero features")
                sys.exit()

            index = index[:col_threshold]
            principle_components = principle_components[:, index]

            img_sub = self.data_dict[keys][2]
            for i in trange(0, self.data_dict[keys][1], desc='data_pre_processing'):
                if keys in self.processed_dataset.keys():
                    self.processed_dataset[keys] = np.dstack((self.processed_dataset[keys],
                                                              principle_components[img_sub*i : img_sub*(i+1)]))
                else:
                    self.processed_dataset[keys] = principle_components[img_sub*i : img_sub*(i+1)]

            print(self.processed_dataset[keys].shape)

        elif transform == 'MDA':
            return

    def train_val_test_split(self, data_name='data', test_ratio=0.8, seed=True, cross_ratio=None):

        if seed:
            np.random.seed(10)

        np.random.shuffle(self.processed_dataset)
        try:
            split_size = int(self.processed_dataset[data_name].shape[0] * test_ratio)
            if split_size == self.processed_dataset[data_name].shape[0]:
                raise ZeroData
        except ZeroData:
            print(f"Data split test_ratio ({test_ratio}) is such that the test split has zero points")
            sys.exit()

        self.train_data = self.processed_dataset[data_name][0: split_size, :, :]
        self.test_data = self.processed_dataset[data_name][split_size:, :, :]

        if cross_ratio is not None:
            try:
                split_size = int(self.train_data.shape[0] * cross_ratio)
                if split_size == self.train_data.shape[0]:
                    raise ZeroData
            except ZeroData:
                print(f"Data split cross_ratio ({cross_ratio}) is such that the val split has zero points")
                sys.exit()

            self.val_data = self.train_data[split_size :, :, :]
            self.train_data = self.train_data[0 : split_size, :, :]


if __name__ == '__main__':
    """ 
    First initialize the dataset object the call load_data.
    """
    data = Dataset()
    data.load_data(transform='PCA', threshold=0.02, data_name='data')
    data.train_test_split()