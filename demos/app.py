import cv2
import numpy as np
from paddleocr import PaddleOCR
import gradio as gr
import matplotlib.pyplot as plt


# Predict function
def predict(input_image):

  model_cfg_path = 'darknet_yolo.cfg'
  model_weights_path = 'model.weights'
  model = cv2.dnn.readNetFromDarknet(cfgFile=model_cfg_path, darknetModel=model_weights_path)

  cv2.imwrite('0.jpg', input_image)

  img = plt.imread('0.jpg')
  height = img.shape[0]
  width = img.shape[1]

  blob_img = cv2.dnn.blobFromImage(image=img, scalefactor=1/255, size=(416, 416), mean=0, swapRB=True)
  model.setInput(blob_img)

  layers = model.getLayerNames()
  output_layers = [layers[i - 1] for i in model.getUnconnectedOutLayers()]
  all_detections = model.forward(output_layers)
  detections = [detection for detections in all_detections for detection in detections if detection[4] * 10000 > 0.33]

  bboxes = []
  class_ids = []
  scores = []

  for detection in detections:
    bbox = detection[:4]
    x_center, y_center, w, h = bbox
    bbox = [int(x_center * width), int(y_center * height), int(w * width), int(h * height)]
    bbox_confidence = detection[4] 
    class_id = np.argmax(detection[5:])
    score = np.amax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)

  if all(item == 0 for item in scores):
    scores[0] = 0.4

  indices =	cv2.dnn.NMSBoxes(bboxes=bboxes, scores=scores, score_threshold=0.33, nms_threshold=0.5)
  x_center, y_center, w, h = bboxes[indices[0]]

  license_plate = img[int(y_center - (h / 2)):int(y_center + (h / 2)), int(x_center - (w / 2)):int(x_center + (w / 2)), :].copy() 
  license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY) 
  _, license_plate_thresh = cv2.threshold(license_plate_gray, 150, 255, cv2.THRESH_BINARY)
  _, license_plate_thresh_2 = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY)

  final_output = {}

  ocr = PaddleOCR(use_angle_cls=True, lang='en')
  result = ocr.ocr(license_plate_thresh, det=False, cls=False)
  if result[0][0][0] == '':
    result = ocr.ocr(license_plate_thresh_2, det=False, cls=False)
  for i in range(len(result)):
    result = result[i]
    final_output = {"".join(e for e in line[0] if e.isalnum()) : f'{line[-1]:.2f}' for line in result}
  
  
  license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)

  return license_plate, final_output

# Gradio Interface
example_list =  [['examples/1.jpg'],
                 ['examples/2.jpg'],
                 ['examples/3.jpg'],
                 ['examples/4.jpg'],
                 ['examples/5.jpg'],
                 ['examples/6.jpg']]

title = "License plate reader 🚌"
description = "[Trained on european car plates] Identifies the license plate, cuts and displays it, and converts it into text. An image with higher resolution and clearer license plate will have a better accuracy."

demo = gr.Interface(fn=predict,
                    inputs='image',
                    outputs=[gr.Image().style(width=512, height=256),
                             gr.Label(label="Prediction - Confidence score")],
                    examples=example_list,
                    title=title,
                    description=description)


demo.launch(debug=False,
            share=False)
