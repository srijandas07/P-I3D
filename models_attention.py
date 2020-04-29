from keras.models import Sequential
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, concatenate, Dense, Flatten, Dropout, Reshape, Input, Add, RepeatVector, Permute
from keras import regularizers
from keras.layers import LSTM, Dense, Activation, Input
from keras.layers import TimeDistributed, GaussianNoise, GaussianDropout, Dropout, Flatten
from keras.models import Model
from keras import backend as K
from i3d_inception import Inception_Inflated3d, conv3d_bn
import keras
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from keras.models import Model, Sequential,  load_model
from keras import layers 
from pi3d import PI3D

f_dept = 832
no_of_p = 0

def inflate_dense(x):
    a = RepeatVector(8*7*7*f_dept)(x)
    a = Permute((2,1), input_shape=(8*7*7*f_dept, no_of_p))(a)
    a = Reshape((no_of_p,8,7,7,f_dept))(a)
    return a

def sum_feature(x):
    return K.sum(x, axis=1)

def concat_feature(x):
    a = Permute((2,3,4,5,1), input_shape=(no_of_p,8,7,7,f_dept))(x)
    a = Reshape((8,7,7,no_of_p*f_dept))(a)
    return a 

def l1_reg(weight_mat):
    return 0.001*K.sum(K.square(weight_mat))

def pi3d_model(fc_main, model_inputs, dataset, protocol, all_models_name=[], mode='sum', dropout_prob=0.0, num_classes=60, sum_idx=0, train_end_to_end=False):
    mode = mode
    all_models_name=all_models_name
    #all_models = {}
    if sum_idx ==0 :
        global f_dept
        f_dept = 1024

    pi3d_interm_outputs = []
    for model_name in all_models_name:
        model = load_model('./weights_optim/{}/weights_{}_{}.hdf5'.format(dataset, model_name, protocol))
        for idx in range(len(model.layers)):
            model.get_layer(index=idx).name=model.layers[idx].name+'_'+model_name

        for l in model.layers:
            l.trainable=train_end_to_end

        model_inputs.append(model.input)
        if sum_idx <= 3 and sum_idx >= 0:
            pi3d_interm_outputs.append(Reshape((1,8,7,7,f_dept))(model.get_layer(index=-46+(2-sum_idx)*20).output))


    x = concatenate(pi3d_interm_outputs, axis=1)
    inflated_fc_main = keras.layers.core.Lambda(inflate_dense, output_shape=(no_of_p, 8, 7, 7, f_dept))(fc_main)
    multiplied_features = keras.layers.Multiply()([inflated_fc_main, x])

    if mode=='sum':
        x = keras.layers.core.Lambda(sum_feature, output_shape=(8, 7, 7, f_dept))(multiplied_features)
    elif mode=='cat':
        x = keras.layers.core.Lambda(concat_feature, output_shape=(8, 7, 7, f_dept*no_of_p))(multiplied_features)

    ##second part of I3D

    if sum_idx==2:
        # Mixed 5b
        branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name=''+'second')

        branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1'+'second')
        branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3'+'second')

        branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1'+'second')
        branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3'+'second')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3'+'second')(x)
        branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1'+'second')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=4,
            name='Mixed_5b'+'second')

    if sum_idx==1 or sum_idx==2:
        # Mixed 5c
        branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1'+'second')

        branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1'+'second')
        branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3'+'second')

        branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1'+'second')
        branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3'+'second')

        branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3'+'second')(x)
        branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1'+'second')

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=4,
            name='Mixed_5c'+'second')

    #Classification block
    x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool'+'second')(x)
    x = Dropout(dropout_prob)(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same',
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1'+'second')

    x = Flatten(name='flatten'+'second')(x)
    predictions = Dense(num_classes, activation='softmax', name='softmax'+'second')(x)
    model = Model(inputs=model_inputs, outputs=predictions, name = 'PI3D')
    
    model_second = Inception_Inflated3d(include_top = True, weights='rgb_imagenet_and_kinetics')
   
    weight_idx_s = -45  + (2-sum_idx)*20
    weight_idx_e = -4
 
    for l_m, l_lh in zip(model.layers[weight_idx_s: weight_idx_e], model_second.layers[weight_idx_s: weight_idx_e]):
        l_m.set_weights(l_lh.get_weights())
        l_m.trainable=True
    
    lstm_weights = "./weights_optim/{}/lstm_model_{}.hdf5".format(dataset, protocol)
    l_model = load_model(lstm_weights, compile=False)

    for idx1 in range(len(model.layers)):
        n1 = model.layers[idx1].name
        if 'lstm' in n1:
            for idx2 in range(len(l_model.layers)):
                n2 = l_model.layers[idx2].name
                if n1==n2:
                    model.layers[idx1].set_weights(l_model.layers[idx2].get_weights())
                    break
    

    return model

def build_model_without_TS(dataset, protocol, n_neuron, n_dropout, batch_size, timesteps, data_dim, num_classes, all_models_name, training_mode='mid', attention_mode='sum', sum_idx=0, train_end_to_end=False) :
    print('Build model...') 
    model_inputs=[]
    x1 = Input(shape=(timesteps, data_dim), name='skeleton_input')                       
    model_inputs.append(x1)

    global no_of_p
    no_of_p = len(all_models_name)

    main_lstm_1 = LSTM(n_neuron, return_sequences=True, trainable=False)(x1)
    main_lstm_2 = LSTM(n_neuron, return_sequences=True, trainable=False)(main_lstm_1)
    main_lstm_3 = LSTM(n_neuron, trainable=False)(main_lstm_2)
    main_lstm_dropped = Dropout(n_dropout, trainable=False, name='droput_1')(main_lstm_3)

    z = Dense(128, activation='tanh', name='z_layer',trainable=False)(main_lstm_dropped)
    fc_main = Dense(no_of_p, activity_regularizer=None, kernel_initializer='zeros', bias_initializer='zeros', activation='softmax',trainable=False, name='dense_1')(z)
    
    model = pi3d_model(fc_main, model_inputs, dataset, protocol, all_models_name, attention_mode, n_dropout, num_classes=num_classes, sum_idx=0, train_end_to_end=False) 
    return model 

def build_model_with_TS(n_neuron, n_dropout, batch_size, timesteps, data_dim, num_classes):
    print('Build model...')         
    model = Sequential()
    model.add(LSTM(n_neuron, return_sequences=True, batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(n_neuron, return_sequences=True))
    model.add(LSTM(n_neuron, return_sequences=True))
    model.add(Dropout(n_dropout))
    model.add(Timedistributed(Dense(num_classes, activation='softmax')))
    return model

