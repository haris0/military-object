import sys
from flask import render_template, request, redirect, Response
from app import app

from core.camera import VideoCamera

@app.route('/live-camera')
def index_live():
    return render_template('live-camera.html')

def gen_live(camera):
    while True:
        frame, success = camera.get_frame()
        if success:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route('/video_feed_live/')
def video_feed_live():
    return Response(gen_live(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')