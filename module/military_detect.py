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
	def __init__(self, out_dir):
		self.net = self.load_model()
		self.Object = self.load_label()
		self.out_dir = out_dir

	def load_label(self):
		labelsPath = './module/obj.names'
		Object = open(labelsPath).read().strip().split("\n")
		return Object

	def load_model(self):
		weightsPath = './module/yolo-tiny-obj.weights'
		configPath = './module/yolo-tiny-obj.cfg'
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

	def draw_box(self, image, predict):
		idxs = cv2.dnn.NMSBoxes(predict['boxes'], predict['conf'], 0.5, 0.3)
		color_dict = {
					'Helikopter Militer' : (103, 58, 183),
					'Jet Militer' : (3, 169, 244),
					'Tank Militer' : (0, 188, 212),
					'Mobil Militer' : (229, 28, 35),
					'Kapal Militer' : (255, 152, 0),
					'Pistol' : (233, 30, 99),
					'Senapan' : (0, 150, 136),
					'Pisau Militer' : (139, 195, 74),
					'Granat' : (205, 220, 57),
					'Tentara' : (158, 158, 158)
			}

		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (predict['boxes'][i][0], predict['boxes'][i][1])
				(w, h) = (predict['boxes'][i][2], predict['boxes'][i][3])
			
				text_color = (255, 255, 255)
				color = color_dict[predict['label'][i]]
				image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				image = cv2.rectangle(image, (x, y+15), (x + 155, y), color, -1)
				text = predict['label'][i] + "({:.2f})".format(predict['conf'][i])
				image = cv2.putText(image, text, (x+2, y+10), cv2.FONT_HERSHEY_PLAIN, 0.8, text_color, 1)
		return image

	def predict_image(self, img_path):
		layer_name = self.get_layer_name()
		image = cv2.imread(img_path)
		Outputs = self.detect_object(image, layer_name)
		boxes, confidences, label = self.get_boxes(Outputs, image)
		predict = {
				'boxes' : boxes,
				'conf'	: confidences,
				'label'	: label
		}
		result = self.draw_box(image, predict)
		filename = get_predict_name(self.out_dir, 'jpg')
		cv2.imwrite(filename, result, [cv2.IMWRITE_JPEG_QUALITY, 100])

		outfile_name = os.path.basename(filename)
		print('Image file finished.')
		return outfile_name
	
	def predict_video(self, video_path):
		
		layer_name = self.get_layer_name()
		cap = cv2.VideoCapture(video_path)  # use 0 for webcam   

		_, frame = cap.read()
		(H, W) = frame.shape[:2]
		
		fourcc = cv2.VideoWriter_fourcc(*'VP90')

		filename = get_predict_name(self.out_dir, 'webm')
		out_vid = cv2.VideoWriter(filename, fourcc, 20.0, (W,H))

		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
	 		
			Outputs = self.detect_object(frame, layer_name)
			boxes, confidences, label = self.get_boxes(Outputs, frame)
			predict = {
					'boxes' : boxes,
					'conf'	: confidences,
					'label'	: label
			}
			result = self.draw_box(frame, predict)

			out_vid.write(result)
		
		cap.release()
		out_vid.release()
		
		outfile_name = os.path.basename(filename)
		print('Video file finished.')
		return outfile_name
	
	def detect_stream(self, vid_path):
		cap = cv2.VideoCapture(vid_path)
		layer_name = self.get_layer_name()
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
				result = self.draw_box(frame, predict)
												
				ret, jpeg = cv2.imencode('.jpg', result)
				frame = jpeg.tobytes()

				yield (b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

		cap.release()