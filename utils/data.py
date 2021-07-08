import numpy as np

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