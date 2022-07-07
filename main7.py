from numpy import save
from veho_det import CarDetector
import cv2
from pathlib import Path
from _data import dict_id_object
import time
import numpy as np
import matplotlib.path as mplPath
import numpy as np




def get_image_with_box(img, bound_boxes):

    for object in bound_boxes:
        #id -> int
        #box -> tuple
        id, box = object
        x, y, w, h = box        

        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
        cv2.putText(img, dict_id_object[id], (x,y), 0, 1, (100, 200, 0), 2)
        cv2.putText(img, "vehicles:" + str(len(bound_boxes)), (20,90), 0, 2.5, (255, 0, 0), 3)
    
    return img

def get_savelocation(raw_img_path):

    #creating the file name to image with boxes
    #using Path
    _file_path = Path(raw_img_path)
    directory = _file_path.parent
    extension = _file_path.suffix
    file_name = _file_path.stem
    
    new_path = directory.joinpath("boxed")

    if not new_path.exists():
        new_path.mkdir()

    save_location = new_path.joinpath(f"{file_name}{extension}")

    print(f"{raw_img_path}\n{save_location}\n")
    return str(save_location)

def _show_both_img(raw_img, detected_img):

    #resize to just fit within screen (no other reason)
    height = 900
    width = int(1920/1080 * height)
    raw_img = cv2.resize(raw_img, (width, height))
    detected_img = cv2.resize(detected_img, (width, height))
    cv2.imshow('original image', raw_img)
    cv2.imshow('detected_image', detected_img)
                
    # cv2.imwrite(save_location, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def main(img_folder_path):
    vd = CarDetector()
    _img_folder_path = Path(img_folder_path)
    
    for img_path in _img_folder_path.iterdir():

        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        print(str(img_path))

        begin = time.perf_counter()
        bound_boxes = vd.detect_vehicles(img)
        end = time.perf_counter()
        print(end-begin)
        
        save_location = get_savelocation(img_path)
        img_with_box = get_image_with_box(img, bound_boxes)
        # _show_both_img(img, img_with_box)
        cv2.imwrite(save_location, img_with_box)


if __name__ == "__main__":
    main("ppimages")
