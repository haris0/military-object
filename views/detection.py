import sys
from app import app
from flask import render_template, redirect, url_for, flash, request, Response
from module.military_detect import yolo_detect, resize_img
# from module.camera import VideoCamera
import os
from werkzeug.utils import secure_filename
import urllib.request
import tldextract
import pytube

UPLOAD_IMG = "./static/upload_img/"
UPLOAD_VID = "./static/upload_vid/"
RESULT_FOLDER = "./static/detect_result/"

app.config['UPLOAD_IMG'] = UPLOAD_IMG
app.config['UPLOAD_VID'] = UPLOAD_VID
app.config['RESULT_FOLDER'] = RESULT_FOLDER

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gp'}

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

def file_type(path):
    filename = path.split('/')[-1]
    if allowed_file_image(filename):
        filetype = 'image'
    elif allowed_file_video(filename):
        filetype = 'video'
    else:
        filetype = 'invalid'
    return filetype

def predict_image_video(path):
    filetype = file_type(path)
    if filetype == 'image':
        resize_img(path)
        model1 = yolo_detect(app.config['RESULT_FOLDER'])
        out_path = model1.predict_image(path)
    elif filetype == 'video':
        out_path = path
    else:
        return redirect(request.url)

    return out_path, filetype

def download(url):
    ext = tldextract.extract(url)
    if ext.domain == 'youtube':
        cleaning_upload_dic(app.config['UPLOAD_VID'])
        print('Youtube')
        path = download_yt(url)
    else:
        cleaning_upload_dic(app.config['UPLOAD_IMG'])
        filename = url.split('/')[-1]
        path = os.path.join(app.config['UPLOAD_IMG'], filename)
        urllib.request.urlretrieve(url, path)

    return path

def download_yt(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    path = video.download(app.config['UPLOAD_VID'])
    
    return path

def save_upload(file):
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        cleaning_upload_dic(app.config['UPLOAD_IMG'])
        path = os.path.join(app.config['UPLOAD_IMG'], filename)
    elif allowed_file_video(filename):
        cleaning_upload_dic(app.config['UPLOAD_VID'])
        path = os.path.join(app.config['UPLOAD_VID'], filename)
    file.save(path)

    return path

@app.route('/')
def root():
    return redirect(url_for('detection'))

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/detection', methods=['GET', 'POST'])
def upload_file():
    cleaning_upload_dic(app.config['RESULT_FOLDER'])
    if request.method == 'POST':
        if "form-submit" in request.form:
            print('URL')
            url = request.form['url_link']
            if url == '':
                print('kosong')
                return render_template('detection.html', error_msg='Isi URL terlebih dahulu!!')
            path = download(url)
            out_path, filetype = predict_image_video(path)
        else:
            print('Upload')
            if 'file' not in request.files:
                print('No file part')
                return render_template('detection.html', error_msg='No File')
            file = request.files['file']
            print(file)
            if file.filename == '':
                print('No selected file')
                return render_template('detection.html', error_msg='No File')
            if file :
                print('Run Upload')
                path = save_upload(file)
                out_path, filetype = predict_image_video(path)

        return render_template('detection.html', title='Home', out_path=out_path, filetype=filetype)

# def gen():
    # while True:
    #     #get camera frame
    #     frame = camera.get_frame()
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    vid_path = request.args.get('out_path')
    model2 = yolo_detect(app.config['RESULT_FOLDER'])
    return Response(model2.detect_stream(vid_path),mimetype='multipart/x-mixed-replace; boundary=frame')