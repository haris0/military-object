import cv2
import numpy as np
import os
from PIL import Image
import random, string

def resize_img(path):
  basewidth = 300
  img = Image.open(path)
  if img.size[1] > img.size[0]:
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  else:
    wpercent = (basewidth/float(img.size[1]))
    wsize = int((float(img.size[0])*float(wpercent)))
    img = img.resize((wsize,basewidth), Image.ANTIALIAS)
  img.save(path, 'JPEG', quality=90)

def randomword(length):
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(length))

def get_predict_name(out_dir, filetype):
	rand = randomword(7)
	if filetype == 'jpg':
		data = 'image'
	else:
		data = 'video'

	return f'{out_dir}{data}_output_{rand}.{filetype}'

class yolo_detect():
	def __init__(self, out_dir, model_path):
		self.out_dir = out_dir
		self.model_path = model_path
		self.net = self.load_model()
		self.Object = self.load_label()

	def load_label(self):
		labelsPath = self.model_path['name']
		Object = open(labelsPath).read().strip().split("\n")
		return Object

	def load_model(self):
		weightsPath = self.model_path['weight']
		configPath = self.model_path['config']
		print("[INFO] loading YOLO from disk...")
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
		
		return net
	
	def get_layer_name(self):
		ln = self.net.getLayerNames()
		ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		return ln
	
	def detect_object(self, image, layer_name):
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		self.net.setInput(blob)
	
		Outputs = self.net.forward(layer_name)
		return Outputs
	
	def get_boxes(self, layerOutputs, image):
		(H, W) = image.shape[:2]
		boxes = []
		confidences = []
		label = []
		for output in layerOutputs:
			for detection in output:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > 0.5:
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					label.append(self.Object[classID])
		return boxes, confidences, label

	def draw_box(self, image, predict, colors):
		idxs = cv2.dnn.NMSBoxes(predict['boxes'], predict['conf'], 0.5, 0.3)

		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (predict['boxes'][i][0], predict['boxes'][i][1])
				(w, h) = (predict['boxes'][i][2], predict['boxes'][i][3])
		
				color = colors[i]
				image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = predict['label'][i] + "({:.2f})".format(predict['conf'][i])
				image = cv2.putText(image, text, (x+2, y+10), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
		return image

	def detect_image(self, img_path):
		layer_name = self.get_layer_name()
		image = cv2.imread(img_path)
		Outputs = self.detect_object(image, layer_name)
		boxes, confidences, label = self.get_boxes(Outputs, image)
		predict = {
				'boxes' : boxes,
				'conf'	: confidences,
				'label'	: label
		}
		colors = np.random.uniform(0, 255, size=(len(self.Object), 3))
		result = self.draw_box(image, predict, colors)
		filename = get_predict_name(self.out_dir, 'jpg')
		cv2.imwrite(filename, result, [cv2.IMWRITE_JPEG_QUALITY, 100])

		outfile_name = os.path.basename(filename)
		print('Image file finished.')
		return outfile_name
	
	def detect_stream(self, vid_path):
		cap = cv2.VideoCapture(vid_path)
		layer_name = self.get_layer_name()
		colors = np.random.uniform(0, 255, size=(len(self.Object), 3))
		while True:
				ret, frame = cap.read()
				if not ret:
						break

				frame=cv2.resize(frame,None,fx=0.7,fy=0.7,
												interpolation=cv2.INTER_AREA)
				
				Outputs = self.detect_object(frame, layer_name)
				boxes, confidences, label = self.get_boxes(Outputs, frame)
				predict = {
						'boxes' : boxes,
						'conf'	: confidences,
						'label'	: label
				}
				result = self.draw_box(frame, predict, colors)
												
				ret, jpeg = cv2.imencode('.jpg', result)
				frame = jpeg.tobytes()

				yield (b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

		cap.release()