from argparse import ArgumentParser
import sys

from NTU_loader import *
from options import *

from keras.models import Sequential
from keras import backend as K
from keras.layers import LSTM, Dense, Activation
from keras.layers import TimeDistributed, GaussianNoise, GaussianDropout, Dropout
from keras.models import Model
import numpy as np
import scipy.io 
import keras
import h5py
import itertools
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import os
from keras.models import Model, Sequential, load_model
from keras.utils import Sequence, multi_gpu_model
from models_attention import build_model_without_TS                                                       
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from multiprocessing import cpu_count


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        directory=self.path
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')
        model_json = self.save_model.to_json()
        with open(self.path + str(self.nb_epoch) + '.json', "w") as json_file:
            json_file.write(model_json)


def custom_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

if __name__ == '__main__':
    args = parse()
    patches_list = [
        'left_hand',
        'right_hand',
        'full_body'
    ]
    
    model_name = '_'.join(patches_list)+'_'+args.training_mode+'_sum_idx'+str(args.sum_idx)+'_'+args.attention_mode+'_split_'+args.split+str(args.train_end_to_end)+args.marker
    
    csvlogger = CSVLogger(model_name+'.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 7)
    optim = keras.optimizers.Adam(lr=0.001, clipnorm=1)
    model = build_model_without_TS(args.dataset, args.protocol, args.n_neuron, args.n_dropout, args.batch_size, args.timesteps, args.data_dim, args.num_classes, patches_list, args.training_mode, args.attention_mode, args.sum_idx, args.train_end_to_end) 
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss = custom_loss, optimizer = optim, metrics = ['accuracy'])
    model.compile(loss=custom_loss, optimizer=optim, metrics=['accuracy'])
    print(model.summary()) 
    
    if training_mode=='mid':
    
        paths = {
            'skeleton': 'data/{}/skeleton/'.format(args.dataset),
            'cnn': 'data/{}/frames/'.format(args.dataset)
        }
    
        train_generator = DataGenerator(paths, patches_list, 'train', batch_size = batch_size)
        val_generator = DataGenerator(paths, patches_list, 'validation', batch_size = batch_size)
        test_generator = DataGenerator(paths, patches_list, 'test', batch_size = batch_size)
    
    elif training_mode=='end':
        dataset_paths=['data/{}/{}/'.format(args.dataset, i) for i in patches_list]
        dataset_splits_path = 'splits/{}/'.format(args.dataset)
    
        paths = {
            'skeleton': 'data/{}/skeleton/'.format(args.dataset),
            'cnn': dataset_paths,
            'split_path': dataset_splits_path
        }
    
        train_generator = DataGeneratorEnd(paths, patches_list, 'train_'+args.protocol, batch_size = batch_size)
        val_generator = DataGeneratorEnd(paths, patches_list, 'validation_'+args.protocol, batch_size = batch_size)
        test_generator = DataGeneratorEnd(paths, patches_list, 'test_'+args.protocol, batch_size = batch_size)
    
    
    model_checkpoint = CustomModelCheckpoint(model, './weights/weights_{}/epoch_'.format(model_name))
    
    parallel_model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        use_multiprocessing=True,
                        epochs=epochs,
                        callbacks = [csvlogger, reduce_lr, model_checkpoint],
                        workers=cpu_count() - 2)

