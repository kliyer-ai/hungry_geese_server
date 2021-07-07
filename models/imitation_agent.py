import pickle
import bz2
import base64
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Activation, Lambda, Add, BatchNormalization, Input
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.utils import to_categorical, Sequence
from glob import glob
import json




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





def create_dataset_from_json(path, json_object=None, standing=0):
    if json_object is None:
        json_open = open(path, 'r')
        json_load = json.load(json_open)
    else:
        json_load = json_object
        
    try:
        winner_index = np.argmax(np.argsort(json_load['rewards']) == 3-standing)

        obses = []
        X = []
        y = []
        actions = {'NORTH':0, 'SOUTH':1, 'WEST':2, 'EAST':3}

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][winner_index]['status'] == 'ACTIVE':
                y_= json_load['steps'][i+1][winner_index]['action']
                if y_ is not None:
                    step = json_load['steps'][i]
                    step[winner_index]['observation']['geese'] = step[0]['observation']['geese']
                    step[winner_index]['observation']['food'] = step[0]['observation']['food']
                    obses.append(step[winner_index]['observation'])
                    y.append(actions[y_])        

        for j in range(len(obses)):
            X_ = make_input(obses[:j+1])
            X_ = centerize(X_)
            X.append(X_)

        X = np.array(X, dtype=np.float32)#[starting_step:]
        y = np.array(y, dtype=np.uint8)#[starting_step:]

        return X, y
    except:
        return 0, 0


class GeeseDataGenerator(Sequence):
        'Generates data for Keras'
        def __init__(self, X_train, y_train, batch_size=128, shuffle=True, sample_weight=None):
            'Initialization'
            self.batch_size = batch_size
            self.X_train = X_train
            self.y_train = y_train
            self.shuffle = shuffle
            self.on_epoch_end()
            self.sample_weight = sample_weight
    
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.ceil(len(self.X_train) / self.batch_size))
    
        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Generate data
            if self.sample_weight is not None:
                X, y, w = self.__data_generation(indexes)
                return X, y, w
            else:
                X, y = self.__data_generation(indexes)
                return X, y
    
        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.X_train))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        
        def __data_generation(self, batch_ids):
            #_, h, w, c = self.X_train.shape
        
            X = self.X_train[batch_ids]
            y = self.y_train[batch_ids]
            if self.sample_weight is not None:
                w = self.sample_weight[batch_ids]
            else:
                pass
            
            series_len = len(X)
            
            # vertical flip
            if np.random.uniform(0,1)>0.5:
                X = np.flip(X, axis=2)
                y = y[:,[0,1,3,2]]
                
            # horizontal flip
            if np.random.uniform(0,1)>0.5:
                X = np.flip(X, axis=1)
                y = y[:,[1,0,2,3]] 
            
            if self.sample_weight is not None:
                return X, y, w
            else:
                return X, y
            


def train_data(model):
    alpha=0.3
    batch_size = 32
    val_size = 0.1
    paths = [path for path in glob('./runs/*.json') if 'info' not in path]

    print(len(paths))

    X_train = []
    y_train = []

    for path in paths:
        print(path)
        X, y = create_dataset_from_json(path, standing=0) # use only winners' moves
        if X is not 0:
            X_train.append(X)
            y_train.append(y)
        
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_train, unique_index = np.unique(X_train, axis=0, return_index=True)
    y_train = y_train[unique_index]

    X_train = np.transpose(X_train, [0, 2, 3, 1])
    y_train = to_categorical(y_train)

    model.compile()
    
    training_generator = GeeseDataGenerator(X_train, (1-alpha)*y_train+0.25*alpha, batch_size=batch_size)
    model.fit(training_generator, epochs=11, batch_size=batch_size)

    return model



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




# RUN THIS WHEN YOU HAVE TO RETRAIN
# model = train_data(model)