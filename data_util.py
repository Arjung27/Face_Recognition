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

    def transform_data(self, transform='PCA'):

        if transform == 'PCA':
            for keys in self.data_dict:
                normalize_data = [self.normalize_image(self.data_dict[keys][0][:,:,j])
                                   for i in range(self.data_dict[keys][1]) for j in range(self.data_dict[keys][2])]

                cov_data = np.matmul(normalize_data.T * normalize_data)

    def normalize_image(self, image):

        mean = np.mean(image)
        std = np.std(image)

        return (image.flatten() - mean) / std

if __name__ == '__main__':
    """ 
    First initialize the dataset object the call load_data. If you want to transform data or 
    get the data in a dictionary call make_data_dict
    """
    data = Dataset()
    data.load_data()