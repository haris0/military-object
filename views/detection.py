import sys
from app import app
from flask import render_template, redirect, url_for, flash, request, Response
from module.military_detect import yolo_detect, resize_img
from module.database import cam_db
import os
from werkzeug.utils import secure_filename
import requests
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

military_model = {
    'name'      : './module/military_weight/obj.names',
    'weight'    : './module/military_weight/yolo-tiny-obj.weights',
    'config'    : './module/military_weight/yolo-tiny-obj.cfg'
}

object_model = {
    'name'      : './module/object_weight/coco.names',
    'weight'    : './module/object_weight/yolov3.weights',
    'config'    : './module/object_weight/yolov3.cfg'
}

db_cam = cam_db()

def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS

def allowed_file_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def cleaning_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
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
        model1 = yolo_detect(app.config['RESULT_FOLDER'], military_model)
        out_path = model1.detect_image(path)
        out_name = "From Image"
    elif filetype == 'video':
        out_path = path
        out_name = "From Video"
    else:
        return redirect(request.url)

    return out_path, out_name, filetype

def download(url):
    ext = tldextract.extract(url)
    if ext.domain == 'youtube':
        try:
            cleaning_dir(app.config['UPLOAD_VID'])
        except:
            pass
        print('Youtube')
        path = download_yt(url)
    else:
        cleaning_dir(app.config['UPLOAD_IMG'])
        filename = url.split('/')[-1]
        path = os.path.join(app.config['UPLOAD_IMG'], filename)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2)',
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                   'Accept-Encoding': 'none',
                   'Accept-Language': 'en-US,en;q=0.8',
                   'Connection': 'keep-alive'}
        r = requests.get(url, stream=True, headers=headers)
        with open(path, "wb") as file:
            file.write(r.content)

    return path

def download_yt(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    path = video.download(app.config['UPLOAD_VID'])
    
    return path

def save_upload(file):
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        cleaning_dir(app.config['UPLOAD_IMG'])
        path = os.path.join(app.config['UPLOAD_IMG'], filename)
    elif allowed_file_video(filename):
        try:
            cleaning_dir(app.config['UPLOAD_VID'])
        except:
            pass
        path = os.path.join(app.config['UPLOAD_VID'], filename)
    file.save(path)

    return path

@app.route('/')
def root():
    return redirect(url_for('detection'))

@app.route('/object_detection')
def detection():
    cam_list = db_cam.getall_cam()
    return render_template('detection.html', title='Home', cam_list=cam_list)

@app.route('/object_detection', methods=['GET', 'POST'])
def upload_file():
    cam_list = db_cam.getall_cam()
    cleaning_dir(app.config['RESULT_FOLDER'])
    if request.method == 'POST':
        if "url-button" in request.form:
            print('URL')
            url = request.form['url_link']
            if url == '':
                print('kosong')
                return render_template('detection.html', error_msg='Isi URL terlebih dahulu!!', cam_list=cam_list)
            path = download(url)
            out_path, out_name, filetype = predict_image_video(path)
        elif "cam-button" in request.form:
            filetype = 'video'
            cam_id = request.form['cam_id']
            out_name = db_cam.get_name_byid(cam_id)

            mode = request.form['mode_type']
            print('mode', mode)
            return render_template('detection.html', title='Home', 
                                    out_path=cam_id, out_name=out_name, 
                                    filetype=filetype, mode=mode, 
                                    cam_list=cam_list)
        else:
            print('Upload')
            if 'file' not in request.files:
                print('No file part')
                return render_template('detection.html', error_msg='No File', cam_list=cam_list)
            file = request.files['file']
            print(file)
            if file.filename == '':
                print('No selected file')
                return render_template('detection.html', error_msg='No File', cam_list=cam_list)
            if file :
                print('Run Upload')
                path = save_upload(file)
                out_path, out_name, filetype = predict_image_video(path)
        print('cam', out_path, filetype)
        return render_template('detection.html', title='Home', 
                                out_path=out_path, out_name=out_name, 
                                filetype=filetype, cam_list=cam_list)

@app.route('/video_feed')
def video_feed():
    vid_path = request.args.get('out_path')
    print('len', len(vid_path))
    if len(vid_path) == 1:
        vid_path = db_cam.get_url_byid(vid_path)
        if vid_path == '0':
            vid_path = int(vid_path)
    print('vid_path', vid_path)
    mode = request.args.get('mode')
    model2 = yolo_detect(app.config['RESULT_FOLDER'], military_model)
    if mode == 'object':
        model2 = yolo_detect(app.config['RESULT_FOLDER'], object_model)
    return Response(model2.detect_stream(vid_path),mimetype='multipart/x-mixed-replace; boundary=frame')