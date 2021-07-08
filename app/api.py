from app import app, active_games
from flask import request
import json
from datetime import datetime
import os
from kaggle_environments import make

from utils.load_imitation_model import load_model
from utils.retrain import train_data

import pickle
import bz2
import base64

@app.route('/play', methods = ['POST'])
def play():
    action = request.json['action']
    game_id = int(request.json['gameId'])
    env, trainer = active_games[game_id]

    # perform the action
    obs, reward, done, info = trainer.step(action)
    resp = {'steps': env.steps, 'done': done}

    state = env.render(mode = 'ansi')
    print(state)
    print(obs)
    print(reward)
    print(info)
    print(done)

    if done:
        with open('runs/{}.json'.format(game_id), 'w') as json_file:
            json.dump(env.toJSON(), json_file)

        playerDidWin = len(obs['geese'][0]) > 0
        resp['playerDidWin'] = playerDidWin

        trainer.reset()
        # free up memory
        del active_games[game_id]

    return json.dumps(resp) 


@app.route('/start', methods = ['POST', 'GET'])
def start():

    # models that the user wants to play agents
    agents = request.json['agents']
    agents = [ './models/' + agent for agent in agents]

    # check runs on disk to obtain next run id
    highest_id = 0
    runs = os.listdir('./runs')
    run_ids = [ int(run.split('.')[0]) for run in runs if run.endswith('.json')]
    
    if len(run_ids) > 0:
        highest_id = max(run_ids)

    # also check current runs in memory
    # that id should be higher because they haven't been written to disk yet
    runs_in_memory = list(active_games.keys())
    if len(runs_in_memory) > 0:
        highest_id = max(highest_id, max(runs_in_memory))

    env = make("hungry_geese", debug=True)
    # None for the user who is playing the game
    trainer = env.train([None] + agents) 

    next_id = highest_id + 1
    active_games[next_id] = (env, trainer)

    print(next_id)
    print(env, trainer)
    return json.dumps({'game_id': next_id}) 


@app.route('/retrain', methods = ['POST'])
def retrain():
    model = load_model('./models/weights')
    model = train_data(model)

    weight_base64 = base64.b64encode(bz2.compress(pickle.dumps(model.get_weights())))
    # w = "weight= %s"%weight_base64

    with open('./models/retrained_weights', 'wb') as f:
        f.write(weight_base64)

    # model.save('./models/retrained_weights.h5')

    return json.dumps({'success': True}) 