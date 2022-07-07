import cv2
import numpy as np

class VehicleDetector:

    def __init__(self):
        # Load Network
        net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)


        # Allow classes containing Vehicles only
        self.classes_allowed = [2, 3, 5, 6, 7]


    def detect_vehicles(self, img, imgsavename):
        # Detect Objects
        vehicles_boxes = []
        # imgsavename = 0
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                # Skip detection with low confidence
                continue

            # if class_id in self.classes_allowed:
            vehicles_boxes.append(box)
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
            cv2.putText(img, "item:" + str(class_id), (x,y), 0, .4, (100, 200, 0), 1)
            cv2.imshow("Cars" , img)
        cv2.putText(img, "vehicles:" + str(len(vehicles_boxes)), (20,30), 0, 1, (100, 100, 0), 2)
        # imgsavename = imgsavename + 1
        cv2.imwrite(str(imgsavename) + ".jpg", img)
        cv2.waitKey(0)
        return vehicles_boxes

