import sys
from flask import render_template
from app import app


@app.route('/')
@app.route('/home')
def index_home():
    return render_template('home.html')