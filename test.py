import numpy as np
import keras
from keras.utils import multi_gpu_model
from keras.models import load_model
import sys
from NTU_loader import *
from models_attention import build_model_without_TS
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from options import *

args = parse()
patches_list = [
    'left_hand',
    'right_hand',
    'full_body'
]

model_name = '_'.join(patches_list)+'_'+args.training_mode+'_sum_idx'+str(args.sum_idx)+'_'+args.attention_mode+'_split_'+args.split+str(args.train_end_to_end)+args.marker
dataset_paths = ['data/{}/{}/'.format(args.dataset, i) for i in patches_list]
dataset_splits_path = 'splits/{}/'.format(args.dataset)

paths = {
    'skeleton': 'data/{}/skeleton/'.format(args.dataset),
    'cnn': dataset_paths,
    'split_path': dataset_splits_path
}
test_generator = DataGeneratorEnd(paths, patches_list, 'test_'+args.protocol, batch_size = batch_size) 
model = build_model_without_TS(args.n_neuron, args.n_dropout, args.batch_size, args.timesteps, args.data_dim, args.num_classes, patches_list, args.training_mode, args.attention_mode, args.sum_idx, args.train_end_to_end)
model.load_weights("weights_"+model_name+"/"+args.model_wt)
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1), metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, clipnorm=1), metrics=['accuracy'])
print(parallel_model.evaluate_generator(generator = test_generator, use_multiprocessing=True, max_queue_size = 48, workers=cpu_count() - 2))

