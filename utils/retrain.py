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
from utils.data import make_input, centerize

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