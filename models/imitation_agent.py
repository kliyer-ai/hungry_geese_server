import pickle
import bz2
import base64
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Activation, Lambda, Add, BatchNormalization, Input
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2, l2


# Neural Network for Hungry Geese
def TorusConv2D(x, ch, kernel, padding="same", strides=1, weight_decay=2e-3):
    x = Lambda(lambda x: K.tile(x, n=(1,3,3,1)), 
               output_shape=lambda input_shape: (None, 3*input_shape[1], 3*input_shape[2], input_shape[3]))(x)
    
    x = Conv2D(ch, kernel, padding=padding, strides=strides,
                      kernel_regularizer=l2(weight_decay))(x)
    
    x = Lambda(lambda x: x[:,int(x.shape[1]/3):2*int(x.shape[1]/3), int(x.shape[2]/3):2*int(x.shape[2]/3),:], 
               output_shape=lambda input_shape: (None, int(input_shape[1]/3), int(input_shape[2]/3), input_shape[3]))(x)
    return x

def conv_bn_relu(x0, ch, kernel, padding="same", strides=1, weight_decay=2e-3, add=False):
    x = TorusConv2D(x0, ch, kernel, padding=padding, strides=strides,
                      weight_decay=weight_decay)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if add:
        x = Add()([x0, x])
    return x

def GeeseNet(input_shape=(7, 11, 17), layers=12, filters=32, weight_decay=2e-3):
    input = Input(input_shape)
    x = conv_bn_relu(input, filters, 3)
    
    for i in range(layers):
        x = conv_bn_relu(x, filters, 3, add=True)
    
    x = GlobalAveragePooling2D()(x)
    
    output = Dense(4, activation='softmax', kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(x)   
    model = Model(input, output)
    #model.compile(optimizer=RadaBelief(learning_rate=1e-3, epsilon=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])    
    
    return model

# Input for Neural Network
def centerize(b):
    dy, dx = np.where(b[0])
    centerize_y = (np.arange(0,7)-3+dy[0])%7
    centerize_x = (np.arange(0,11)-5+dx[0])%11
    
    b = b[:, centerize_y,:]
    b = b[:, :,centerize_x]
    
    return b

def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs['index']) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs['index']) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs['index']) % 4, pos] = 1
            
    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[12 + (p - obs['index']) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1
        
    b = b.reshape(-1, 7, 11)
    b = centerize(b) # Where to place the head is arbiterary dicision.

    return b


# Load Keras Model
weight = None
with open('./models/weight', 'rb') as f:
    weight = f.read()


model = GeeseNet(input_shape=(7, 11, 17), layers=12, filters=32, weight_decay=1e-7)
model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weight))))

# Main Function of Agent

obses = []

def agent(obs_dict, config_dict):
    obses.append(obs_dict)

    X_test = make_input(obses)
    X_test = np.transpose(X_test, (1,2,0))
    X_test = X_test.reshape(-1,7,11,17) # channel last.
    
    # avoid suicide
    obstacles = X_test[:,:,:,[8,9,10,11,12]].max(axis=3) - X_test[:,:,:,[4,5,6,7]].max(axis=3) # body + opposite_side - my tail
    obstacles = np.array([obstacles[0,2,5], obstacles[0,4,5], obstacles[0,3,4], obstacles[0,3,6]])
    
    y_pred = model.predict(X_test) - obstacles

    
    
    actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    return actions[np.argmax(y_pred)]