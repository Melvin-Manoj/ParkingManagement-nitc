import cv2
import numpy as np

class CarDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)

        self.classes_allowed = [1,2,3,5,7]

    def detect_vehicles(self, img):
        # Detect Objects
        vehicles_boxes = []

        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                # Skip detection with low confidence
                continue
            if class_id in self.classes_allowed:
                vehicles_boxes.append((class_id[0], box))
            # print("in detect vehicles" + str(type(class_id)))
            # print(class_id)
            # print(class_id.size)
            # print(class_id[0])
        return vehicles_boxes