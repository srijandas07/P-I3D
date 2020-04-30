import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import keras
import h5py
import scipy
import glob
from keras.utils import Sequence, to_categorical
from random import sample, randint, shuffle
from keras.utils import Sequence, to_categorical
import cv2 


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, patches_list, mode, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.path_skeleton = paths['skeleton']
        self.path_cnn = paths['cnn']
        self.patches_list = patches_list
        self.num_classes = 60
        self.step = 20
        self.dim = 150 ##for two skeletons in a single frame
        self.mode = mode
        self.hdf5_paths = [self.path_cnn+i+'/'+self.mode+'.hdf5' for i in self.patches_list]
        self.hdf5_files = [h5py.File(i,'r') for i in self.hdf5_paths]
        self.feature_data_cnn_list = [np.array(i[self.mode]) for i in self.hdf5_files]
        self.labels_data = np.array(self.hdf5_files[0]['labels'])
        self.feature_data_skeleton = scipy.io.loadmat(self.path_skeleton+mode+'.mat')
        self.feature_data_skeleton = self.feature_data_skeleton['data'][0]        


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.labels_data.shape[0] / self.batch_size))

    def __getitem__(self, idx):

        i_s = idx * self.batch_size
        i_e = (idx + 1) * self.batch_size

        x_data_cnn = self._get_data_cnn(i_s, i_e)
        x_data_skeleton = self._get_data_skeleton(i_s, i_e)

        y_data = np.array(self.labels_data[i_s:i_e]) - 1
        y_data = to_categorical(y_data, num_classes = self.num_classes)

        return [x_data_skeleton, x_data_cnn], [y_data, y_data]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        idx = np.random.permutation(self.feature_data_cnn_list[0].shape[0])
        self.feature_data_cnn_list = [d[idx] for d in self.feature_data_cnn_list]
        self.feature_data_skeleton = self.feature_data_skeleton[idx]
        self.labels_data = self.labels_data[idx]
        pass
      

    def _get_data_skeleton(self, i_s, i_e):

        x_data_skeleton = self.feature_data_skeleton[i_s: i_e]
        
        for i in range(len(x_data_skeleton)):
            unpadded_file = x_data_skeleton[i] 
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1 
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            x_data_skeleton[i] = np.squeeze(sampled_file)
        
        x_data_skeleton = np.asarray(list(x_data_skeleton))
        return x_data_skeleton

    def _get_data_cnn(self, i_s, i_e):

        x_data = []
        for item in self.feature_data_cnn_list:
            x = np.array(item[i_s:i_e])
            x = x.reshape((x.shape[0],7,1024))
            x_data.append(x)
        
        x = np.concatenate(x_data, axis=1)
        x_shape = (x.shape[0], len(x_data), 7, 1024)
        x = x.reshape((x_shape))
        return x


class DataGeneratorEnd(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, patches_list, mode, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.path_skeleton = paths['skeleton']
        self.path_cnn = paths['cnn']
        self.files = [i.strip() for i in open(paths['split_path']+mode+'.txt').readlines()]
        self.patches_list = patches_list
        self.num_classes = 60
        self.stack_size = 64
        self.stride = 2
        self.step = 20
        self.dim = 150 ##for two skeletons in a single frame
        self.mode = mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):

        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]

        x_data = []
        x_data_cnn = self._get_data_cnn(batch)
        x_data_skeleton = self._get_data_skeleton(batch)
        x_data.append(x_data_skeleton)
        x_data.extend(x_data_cnn)

        y_data = np.array([int(i[-3:]) for i in batch]) - 1
        y_data = to_categorical(y_data, num_classes = self.num_classes)

        return x_data, y_data

    def on_epoch_end(self):
        shuffle(self.files)
        pass

    def _get_data_skeleton(self, list_IDs_temp):

        # Initialization
        X = np.empty((self.batch_size, self.step, self.dim))
  
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X[i,] = np.squeeze(sampled_file)
            X = np.asarray(X)

        return X

    def _get_data_cnn(self, batch):

        x_train_multi = []
        for dataset_path in self.path_cnn:
            x_train = [self._get_video(i, dataset_path) for i in batch]
            x_train = np.array(x_train, np.float32)
            x_train /= 127.5
            x_train -= 1
            x_train_multi.append(x_train)

        return x_train_multi

    def _get_video(self, vid_name, dataset_path):
        images = glob.glob(dataset_path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr


