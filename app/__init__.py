from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# ============================================
# needed for the game
from kaggle_environments import make

env = make("hungry_geese", debug=True)
trainer = env.train([None, "imitation_agent.py"])
obs = trainer.reset()

def perform(action):
    obs, reward, done, info = trainer.step(action)
    print(obs)
    print(reward)
    print(info)
    print(done)
    if done:
        obs = trainer.reset()
# ============================================

from app import views, api


