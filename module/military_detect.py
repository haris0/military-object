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

def get_recent_img(path):
  img_list = os.listdir(path)
  img_list = sorted(img_list, reverse=True)
  if len(img_list) > 6:
    os.remove(os.path.join(path, img_list[-1]))
    img_list = img_list[:6]
    for img in img_list:
      old = os.path.join(path, img)
      count = img.split('_')[0]
      name = '_'.join(i for i in img.split('_')[1:]) 
      count = (int(count)-1)
      new = os.path.join(path, f'{count}_{name}')
      os.rename(old,new)
    img_list = os.listdir(path)
    img_list = sorted(img_list, reverse=True)

  return img_list[:6]

def randomword(length):
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(length))

def deg_predict_name(out_dir):
  list_img = get_recent_img(out_dir)
  if not list_img:
    count = '1'
  else:
    last_img = list_img[0]
    count = int(last_img.split('_')[0]) + 1
  rand = randomword(7)
  return f'{out_dir}{count}_result_output_{rand}.jpg'

class yolo_detect:
  def __init__(self, out_dir):
    self.net = self.load_model()
    self.out_dir = out_dir

  def load_label(self):
    labelsPath = './module/obj.names'
    self.Object = open(labelsPath).read().strip().split("\n")

  def load_model(self):
    weightsPath = './module/yolo-tiny-obj.weights'
    configPath = './module/yolo-tiny-obj.cfg'

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    return net

  def predict_img(self, img_path):
    self.load_label()
    self.load_model()

    self.image = cv2.imread(img_path)
    (H, W) = self.image.shape[:2]

    ln = self.net.getLayerNames()
    ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    self.net.setInput(blob)
    layerOutputs = self.net.forward(ln)

    self.boxes = []
    self.confidences = []
    self.label = []

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

          self.boxes.append([x, y, int(width), int(height)])
          self.confidences.append(float(confidence))
          print(classID)
          self.label.append(self.Object[classID])

    self.draw_labeled_image()
    outfile_name = os.path.basename(self.result_dir)

    return outfile_name

  def draw_labeled_image(self):
    idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.3)
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
        (x, y) = (self.boxes[i][0], self.boxes[i][1])
        (w, h) = (self.boxes[i][2], self.boxes[i][3])

        text_color = (255, 255, 255)
        color = color_dict[self.label[i]]
        self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
        self.image = cv2.rectangle(self.image, (x, y+15), (x + 155, y), color, -1)
        text = self.label[i] + "({:.2f})".format(self.confidences[i])
        self.image = cv2.putText(self.image, text, (x+2, y+10), cv2.FONT_HERSHEY_PLAIN, 0.8, text_color, 1)
    
    self.result_dir = deg_predict_name(self.out_dir)
    print(self.result_dir)
    cv2.imwrite(self.result_dir, self.image, [cv2.IMWRITE_JPEG_QUALITY, 100])