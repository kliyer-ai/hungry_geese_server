from app import app, env, trainer
from flask import request
import json
from datetime import datetime

@app.route('/play', methods = ['POST', 'GET'])
def play():
    action = 'WEST'
    # GET is just for easy testing/debugging
    if request.method == 'POST':
        action = request.json['action']

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
        with open('runs/run_{}.json'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), 'w') as json_file:
            json.dump(run_dict, json_file)

        obs = trainer.reset()

    
    return json.dumps({'steps': env.steps, 'done': done}) 