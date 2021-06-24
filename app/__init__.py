from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# ============================================
# maps from game id to (env, trainer)
active_games = {}



def perform(action, game_id):
    env, trainer = active_games[game_id]
    obs, reward, done, info = trainer.step(action)
    print(obs)
    print(reward)
    print(info)
    print(done)
    if done:
        obs = trainer.reset()
# ============================================

from app import views, api


