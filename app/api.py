from app import app, env, trainer
from flask import request
import json

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
        obs = trainer.reset()

    
    return json.dumps({'steps': env.steps, 'done': done}) 