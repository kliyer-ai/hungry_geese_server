from app import app
from flask import render_template

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/game')
def game():
    return render_template('game.html')