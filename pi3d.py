from keras.models import Model
from keras.layers import concatenate
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.models import Model, Sequential,  load_model
from keras import backend as K
from keras import layers 
import keras

class PI3D:
    def __init__(self, num_classes, train_end_to_end):
        self.num_classes = num_classes
        self.train_end_to_end = train_end_to_end

    def sum_feature(self, x):
        #return x[0]*0.5 + x[1]*0.5
        return K.sum(x, axis=1)

    def pi3d_model(self, all_models_name, mode, dropout_prob, sum_idx):
        self.mode = mode
        self.all_models_name=all_models_name
        all_models = {}
        pi3d_inputs = []
        pi3d_interm_outputs = []
        for model_name in all_models_name:
            model = load_model('./weights_optim/' + model_name + '/weights.hdf5')
            for idx in range(len(model.layers)):
                model.get_layer(index=idx).name=model.layers[idx].name+'_'+model_name
                
                if sum_idx <= 3 and sum_idx >= 0: 
                    all_models[model_name] = Model(inputs=model.input, outputs=model.get_layer(index=-46 + (2-sum_idx)*20).output) ##max_pooling3d_11 (8,7,7,832)
            
            for l in all_models[model_name].layers:
                l.trainable=self.train_end_to_end

            pi3d_inputs.append(all_models[model_name].input)
            pi3d_interm_outputs.append(all_models[model_name].output)
            #pi3d_interm_outputs.append(Reshape((1,8,7,7,832))(all_models[model_name].output))
        
        if self.mode=='sum':
            #keras.core.LambdaMerge([model0, model1], lambda inputs: p0*inputs[0]+p1*inputs[1]))
            #x = concatenate(pi3d_interm_outputs, axis=1)
            x = Add()(pi3d_interm_outputs)
            #x = keras.layers.core.Lambda(self.sum_feature, output_shape=(8, 7, 7, 832))(x)
        elif self.mode=='cat': 
            x = concatenate(pi3d_interm_outputs)
            #pass
        elif self.mode=='single':
            x = pi3d_interm_outputs[0] 
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

        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool'+'second')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, self.num_classes, 1, 1, 1, padding='same', 
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1'+'second')
 
        x = Flatten(name='flatten'+'second')(x)
        x = Dense(self.num_classes, activation='softmax', name='softmax'+'second')(x)
        model = Model(inputs=pi3d_inputs, outputs=x, name = 'PI3D')

        return model   

    def initialize_weights(self, all_models_name, mode, dropout_prob, sum_idx):
        model = self.pi3d_model(all_models_name, mode, dropout_prob, sum_idx)
        #model_second = load_model('/data/stars/user/achaudha/ACCV_2018/I3D_experiments_all_patches/new_model/weights_optim/' + 'left_hand' + '/weights.hdf5')
        #model_second = load_model('/data/stars/user/achaudha/ACCV_2018/I3D_experiments_all_patches/new_model/weights_optim/' + 'i3d' + '/weights.hdf5')
        model_second = Inception_Inflated3d(include_top = True, weights='rgb_imagenet_and_kinetics')
        #pi3d = PI3D2(self.num_classes)
        #model_second = pi3d.initialize_weights(all_models_name, mode, dropout_prob)
        #model_second.load_weights('/data/stars/user/achaudha/ACCV_2018/PI3D_a/weights_pi3d_left_hand_full_body_sum1/epoch8.hdf5') 
        weight_idx_s = -45  + (2-sum_idx)*20
        weight_idx_e = -4
        for l_m, l_lh in zip(model.layers[weight_idx_s: weight_idx_e], model_second.layers[weight_idx_s: weight_idx_e]):
            l_m.set_weights(l_lh.get_weights())

        return model




