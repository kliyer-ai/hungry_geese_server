from app import app, active_games
from flask import request
import json
from datetime import datetime
import os
from kaggle_environments import make

@app.route('/play', methods = ['POST'])
def play():

    action = request.json['action']
    game_id = int(request.json['gameId'])
    
    env, trainer = active_games[game_id]

    # perform the action
    obs, reward, done, info = trainer.step(action)
    state = env.render(mode = 'ansi')

    print(state)
    print(obs)
    print(reward)
    print(info)
    print(done)
    if done:
        # save game as json for future learning
        run_dict = {
            'steps': env.steps,
            'configuration': env.configuration
        }
        with open('runs/{}.json'.format(game_id), 'w') as json_file:
            json.dump(run_dict, json_file)

        obs = trainer.reset()

    
    return json.dumps({'steps': env.steps, 'done': done}) 


@app.route('/start', methods = ['POST', 'GET'])
def start():

    runs = os.listdir('./runs')
    run_ids = [ int(run.split('.')[0]) for run in runs if run.endswith('.json')]

    next_id = 0
    if len(run_ids) > 0:
        next_id = max(run_ids) + 1

    env = make("hungry_geese", debug=True)
    trainer = env.train([None, "./models/imitation_agent.py"])

    active_games[next_id] = (env, trainer)

    print(next_id)
    return json.dumps({'game_id': next_id}) 