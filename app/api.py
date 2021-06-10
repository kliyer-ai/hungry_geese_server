from app import app, env, perform
from flask import request
import json

@app.route('/play', methods = ['POST', 'GET'])
def play():
    action = 'WEST'
    # GET is just for easy testing/debugging
    if request.method == 'POST':
        action = request.json['action']

    perform(action)
    state = env.render(mode = 'ansi')
    print(state)
    return json.dumps(env.steps) 