
import tensorflow as tf
from utils.data import make_input
import numpy as np

model = tf.keras.models.load_model('./models/my_model.h5')

obses = []

def agent(obs_dict, config_dict):
    obses.append(obs_dict)

    X_test = make_input(obses)
    X_test = X_test.reshape(-1,7,11,17) # channel last.
    
    y_pred = model.predict(X_test) 

    actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    return actions[np.argmax(y_pred)]