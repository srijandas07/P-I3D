import numpy as np
import keras
from keras.utils import multi_gpu_model
from keras.models import load_model
import sys
from smarthomes_skeleton_CNN_loader import *
from models_attention import build_model_without_TS
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

def _name_to_int(name, protocol):
    if protocol=='CS':
        integer=0
        if name=="Cook.Cleandishes":
            integer=1
        elif name=="Cook.Cleanup":
            integer=2
        elif name=="Cook.Cut":
            integer=3
        elif name=="Cook.Stir":
            integer=4
        elif name=="Cook.Usestove":
            integer=5
        elif name=="Cutbread":
            integer=6
        elif name=="Drink.Frombottle":
            integer=7
        elif name=="Drink.Fromcan":
            integer=8
        elif name=="Drink.Fromcup":
            integer=9
        elif name=="Drink.Fromglass":
            integer=10
        elif name=="Eat.Attable":
            integer=11
        elif name=="Eat.Snack":
            integer=12
        elif name=="Enter":
            integer=13
        elif name=="Getup":
            integer=14
        elif name=="Laydown":
            integer=15
        elif name=="Leave":
            integer=16
        elif name=="Makecoffee.Pourgrains":
            integer=17
        elif name=="Makecoffee.Pourwater":
            integer=18
        elif name=="Maketea.Boilwater":
            integer=19
        elif name=="Maketea.Insertteabag":
            integer=20
        elif name=="Pour.Frombottle":
            integer=21
        elif name=="Pour.Fromcan":
            integer=22
        elif name=="Pour.Fromcup":
            integer=23
        elif name=="Pour.Fromkettle":
            integer=24
        elif name=="Readbook":
            integer=25
        elif name=="Sitdown":
            integer=26
        elif name=="Takepills":
            integer=27
        elif name=="Uselaptop":
            integer=28
        elif name=="Usetablet":
            integer=29
        elif name=="Usetelephone":
            integer=30
        elif name=="Walk":
            integer=31
        elif name=="WatchTV":
            integer=32
    else:
        if name=="Cutbread":
            integer=1
        elif name=="Drink.Frombottle":
            integer=2
        elif name=="Drink.Fromcan":
            integer=3
        elif name=="Drink.Fromcup":
            integer=4
        elif name=="Drink.Fromglass":
            integer=5
        elif name=="Eat.Attable":
            integer=6
        elif name=="Eat.Snack":
            integer=7
        elif name=="Enter":
            integer=8
        elif name=="Getup":
            integer=9
        elif name=="Leave":
            integer=10
        elif name=="Pour.Frombottle":
            integer=11
        elif name=="Pour.Fromcan":
            integer=12
        elif name=="Readbook":
            integer=13
        elif name=="Sitdown":
            integer=14
        elif name=="Takepills":
            integer=15
        elif name=="Uselaptop":
            integer=16
        elif name=="Usetablet":
            integer=17
        elif name=="Usetelephone":
            integer=18
        elif name=="Walk":
            integer=19
    return integer

model_wt = sys.argv[1]
batch_size = 4
split='CS'
data_dim = 39
if split == 'CS':
    num_classes = 32
    n_neuron = 512
else:
    num_classes = 19
    n_neuron = 128
batch_size = 4
n_dropout = 0.5
training_mode = 'end' ## 'mid' or 'end', 'mid' is used if mid level features are prestored.
attention_mode = 'sum' ## 'sum' or 'cat'
sum_idx = 0
train_end_to_end = False
marker='_smarthomes_reg_no2'
timesteps=5
patches_list = [
    'left_hand',
    'right_hand',
    'full_body'
]

model_name = '_'.join(patches_list)+'_'+training_mode+'_sum_idx'+str(sum_idx)+'_'+attention_mode+'_split_'+split+str(train_end_to_end)+marker
dataset_paths=["./left_hand2/", "./right_hand2/", "/data/stars/user/rdai/smarthomes/Blurred_smarthome_clipped_SSD/"]
dataset_splits_path = "/data/stars/user/sdas/smarthomes_data/splits/"

paths = {
    'skeleton': '/data/stars/user/rdai/smarthomes/smarthome_clipped_npz/',
    'cnn': dataset_paths,
    'split_path': dataset_splits_path
}
test_generator = DataGeneratorEnd(paths, patches_list, 'test_'+split, batch_size = batch_size) 
model = build_model_without_TS(n_neuron, n_dropout, batch_size, timesteps, data_dim, num_classes, patches_list, training_mode, attention_mode, sum_idx, train_end_to_end)
model.load_weights("weights_"+model_name+"/"+model_wt)
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1), metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, clipnorm=1), metrics=['accuracy'])
prediction_result = parallel_model.predict_generator(generator = test_generator, use_multiprocessing=True, max_queue_size = 48, workers=cpu_count() - 2)

y_true=np.array([_name_to_int(i.split('_')[0], split) for i in open(dataset_splits_path+'test_'+split+'.txt').readlines()]) - 1
y_pred=np.argmax(prediction_result, axis=-1)
y_pred=np.array(y_pred,np.int)
print(np.min(y_pred), np.max(y_pred))
print(np.min(y_true), np.max(y_true))
print(accuracy_score(y_true[:len(y_pred)], y_pred))
print(balanced_accuracy_score(y_true[:len(y_pred)], y_pred))
