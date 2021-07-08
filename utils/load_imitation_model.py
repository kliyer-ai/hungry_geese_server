import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Activation, Lambda, Add, BatchNormalization, Input
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2, l2
import pickle
import bz2
import base64


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


def load_model(path):
    weight = None
    with open(path, 'rb') as f:
        weight = f.read()


    model = GeeseNet(input_shape=(7, 11, 17), layers=12, filters=32, weight_decay=1e-7)
    model.set_weights(pickle.loads(bz2.decompress(base64.b64decode(weight))))

    return model