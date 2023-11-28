import json
import cv2
import imutils
import numpy as np
from datetime import datetime
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='peoples')

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2


def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)
    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    return results


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model_det = cv2.dnn.readNetFromDarknet(config_path, weights_path)
'''
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''
layer_name = model_det.getLayerNames()
layer_name = [layer_name[i - 1] for i in model_det.getUnconnectedOutLayers()]
cap = cv2.VideoCapture("your_video.avi")

writer = None
fps = 30
current_frame = 0

while True:
    rmq_people_message = {}

    (grabbed, image) = cap.read()
    current_frame += 1
    rmq_people_message["frame"] = current_frame
    if not grabbed:
        break
    image = imutils.resize(image, width=700)
    results = pedestrian_detection(image, model_det, layer_name,
                                   personidz=LABELS.index("person"))

    rmq_people_message["n_people"] = len(results)
    rmq_people_message["people"] = {}
    for i, res in enumerate(results):

        (conf, bbox, centroid) = res
        (startX, startY, endX, endY) = bbox
        person_image = image[startY:endY, startX:endX]
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        if person_image.size == 0:
            continue
        if current_frame % fps == 0:
            print(f"Saved person ({current_frame // fps} s)")
            # cv2.imwrite("people/person_{}.jpg".format(datetime.now().timestamp()), person_image)
            person_image_path = "/Volumes/RAMDisk/all_peoples/person_{}.png".format(datetime.now().timestamp())
            cv2.imwrite(person_image_path, person_image)

            current_person_data = {"RAM_path": str(person_image_path),
                                   "x0": str(startX),
                                   "y0": str(startY),
                                   "x1": str(endX),
                                   "y1": str(endY),
                                   "timestamp": str(datetime.timestamp(datetime.now())),
                                   "proba": str(conf),
                                   "fps": str(fps),
                                   }

            rmq_people_message["people"][i] = current_person_data

            channel.basic_publish(exchange='', routing_key='peoples',
                                  body=json.dumps(current_person_data))

channel.close()
connection.close()
cap.release()
cv2.destroyAllWindows()
