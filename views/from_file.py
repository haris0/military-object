import os
import sys
from flask import render_template, request, redirect, Response, flash, url_for
from werkzeug.utils import secure_filename

from app import app

from core.camera import VideoCamera

UPLOAD_FOLDER = 'temp/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/from-file', methods=["GET", "POST"])
def index_file():
    if request.method == 'POST':
        file = request.files['video']
        if 'video' not in request.files:
            flash('No file part')
        elif file.filename == '':
            flash('No selected file')
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('from-file.html', filename=filename)
        else:
            flash('Extension not valid')

        return redirect(request.url)
    else:
        return render_template('from-file.html')

def gen_file(camera):
    while True:
        frame, success = camera.get_frame()
        if success:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route('/video_feed_file/<filename>')
def video_feed_file(filename):
    url_video = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(gen_file(VideoCamera(url_video)), mimetype='multipart/x-mixed-replace; boundary=frame')