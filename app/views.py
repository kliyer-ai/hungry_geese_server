from app import app
from flask import render_template
import os

@app.route('/')
def index():
    agents = os.listdir('./models')
    agents = [ agent for agent in agents if agent.endswith('.py')]
    return render_template('start.html', **{'agents': agents} )

@app.route('/game')
def game():
    return render_template('game.html')