import sys
from flask import render_template, redirect, url_for
from app import app


@app.route('/')
def root():
    return redirect(url_for('detection'))

@app.route('/detection')
def detection():
    return render_template('detection.html')