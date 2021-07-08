from app import app
from flask import render_template, send_file
import os
import shutil

@app.route('/')
def index():
    agents = os.listdir('./models')
    agents = [ agent for agent in agents if agent.endswith('.py')]
    return render_template('start.html', **{'agents': agents} )

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/game/over')
def over():
    return render_template('over.html')

@app.route('/download')
def download():
    shutil.make_archive('runs', 'zip', './runs')
    return send_file('../runs.zip', as_attachment=True)
