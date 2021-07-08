import numpy as np
from utils.load_imitation_model import load_model
from utils.data import make_input




model = load_model('./models/weight')

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




# RUN THIS WHEN YOU HAVE TO RETRAIN
# model = train_data(model)