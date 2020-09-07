import sys
from app import app
from flask import render_template, redirect, url_for, flash, request, Response
from module.military_detect import yolo_detect, resize_img
# from module.camera import VideoCamera
import os
from werkzeug.utils import secure_filename
import urllib.request

UPLOAD_FOLDER = "./static/upload/"
RESULT_FOLDER = ".static/detect_result"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gp'}

ym = yolo_detect(app.config['RESULT_FOLDER'])

def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS

def allowed_file_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def cleaning_upload_dic(path):
    if not os.listdir(path):
        print('Folder Empty')
    else:
        filelist = [ f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f))

def predict_image_videe(url):
    filename = url.split('/')[-1]
    if allowed_file_image(filename):
        path = download(url)
        resize_img(path)
        out_path = ym.predict_image(path)

    elif allowed_file_video(filename):
        path = download(url)

    return out_path

def download(url):
    filename = url.split('/')[-1]
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    urllib.request.urlretrieve(url, path)
    return path

@app.route('/')
def root():
    return redirect(url_for('detection'))

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/detection', methods=['GET', 'POST'])
def upload_file():
    cleaning_upload_dic(app.config['UPLOAD_FOLDER'])
    cleaning_upload_dic(app.config['RESULT_FOLDER'])
    if request.method == 'POST':
        if "form-submit" in request.form:
            print('URL')
            url = request.form['url_link']
            out_img = predict_image_videe(url)
        else:
            print('Upload')
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url, error_msg='No File')
            file = request.files['file']
            print(file)
            if file.filename == '':
                print('No selected file')
                return redirect(request.url, error_msg='No File')
            if file and allowed_file_image(file.filename):
                print('Run Upload')
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                resize_img(path)
                out_img = ym.predict_image(path)

        return render_template('detection.html', title='Home', out_img=out_img)

# def gen(camera):
#     while True:
#         #get camera frame
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')