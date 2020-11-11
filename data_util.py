import numpy as np
from scipy.io import loadmat
import os

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

    def load_data(self):

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

        elif self.task_id == 2:
            self.data = loadmat(os.path.join(self.data_path, 'data.mat'))['face']

        self.std_data = {}

        for i in range(self.data_sub):
            self.std_data[i] = \
                self.data[:, :, 3*i : 3*(i+1)]

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

    def transform_data(self, data_name='data', transform='PCA', threshold=0.8):

        self.make_data_dict()
        if transform == 'PCA':
            keys = data_name
            print(keys)
            for i in range(self.data_dict[keys][1]):
                print(i)
                normalize_data = [self.normalize_image(self.data_dict[keys][0][i][:,:,j])
                                   for j in range(self.data_dict[keys][2])]
                normalize_data = np.array(normalize_data, dtype=np.float64)

                cov_data = np.matmul(normalize_data.T, normalize_data)
                eig_value, eig_vector = np.linalg.eig(cov_data)

                # Sorting eigen value and corresponding eig vector in descending order
                index = np.argsort(eig_value)
                eig_value_sort = eig_value[index]
                eig_value_sort = eig_value_sort[::-1]
                eig_vector_sort = eig_vector[:, index]
                eig_vector_sort = np.flip(eig_vector_sort, axis=-1)

                # Taking only real values
                eig_value_sort = eig_value_sort.real
                eig_vector_sort = eig_vector_sort.real

                principle_componets = np.matmul(normalize_data, eig_vector_sort)
                variance = np.std(principle_componets, axis=0)
                index = np.argsort(variance)
                col_threshold = int(eig_vector_sort.shape[-1] * threshold)
                index = index[-col_threshold:]
                eig_vector_sort = eig_vector_sort[:, index]

                if keys in self.processed_dataset.keys():
                    self.processed_dataset[keys] = np.dstack((self.processed_dataset[keys], eig_vector_sort))
                else:
                    self.processed_dataset[keys] = eig_vector_sort

        elif transform == 'MDA':
            return

if __name__ == '__main__':
    """ 
    First initialize the dataset object the call load_data.
    """
    data = Dataset()
    data.load_data()
    data.transform_data(data_name='data')