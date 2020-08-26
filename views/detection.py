import sys
from app import app
from flask import render_template, redirect, url_for, flash, request, redirect
from module.military_detect import yolo_detect, resize_img, get_recent_img
import os
from werkzeug.utils import secure_filename
import urllib.request

UPLOAD_FOLDER = "./static/upload/"
RESULT_FOLDER_IMAGE = "./static/detect_image/"
RESULT_FOLDER_VIDEO = "./static/detect_video/"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER_IMAGE'] = RESULT_FOLDER_IMAGE
app.config['RESULT_FOLDER_VIDEO'] = RESULT_FOLDER_VIDEO

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gp'}

ym = yolo_detect(app.config['RESULT_FOLDER_IMAGE'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS

def cleaning_upload_dic(path):
    if not os.listdir(path):
        print('Folder Empty')
    else:
        filelist = [ f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f))

@app.route('/')
def root():
    return redirect(url_for('detection'))

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/detection', methods=['GET', 'POST'])
def upload_file():
    cleaning_upload_dic(app.config['UPLOAD_FOLDER'])
    cleaning_upload_dic(app.config['RESULT_FOLDER_IMAGE'])
    if request.method == 'POST':
        if "form-submit" in request.form:
            print('URL')
            url = request.form['url_link']
            filename = url.split('/')[-1]
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            urllib.request.urlretrieve(url, path)
            resize_img(path)
            out_img = ym.predict_img(path)
        else:
            print('Upload')
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url)
            file = request.files['file']
            print(file)
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                print('Run Upload')
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                resize_img(path)
                out_img = ym.predict_img(path)

        return render_template('detection.html', title='Home', out_img=out_img)