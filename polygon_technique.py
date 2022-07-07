from itertools import count
from numpy import poly, save
from veho_det import CarDetector
import cv2
from pathlib import Path
from _data import dict_id_object, atm_polygon
import time
import numpy as np
import matplotlib.path as mplPath
import numpy as np
from _helper import find_center
from main import get_image_with_box


#polygon_corners should be an np array of size=(n,2)
def is_point_inside(polygon_corners, point):
    poly_path = mplPath.Path(polygon_corners)
    return poly_path.contains_point(point)

def draw_polygon(polygon_corners, image):

    polygon_corners = polygon_corners.reshape((-1,2))
    # Blue color in BGR
    # Line thickness of 2 px
    image = cv2.polylines(image, [polygon_corners],
                        isClosed = False, color = (255,0,0), 
                        thickness = 2)
    return image


def count_draw_cars(img_path):
    img = cv2.imread(img_path)
    
    img = draw_polygon(atm_polygon,  img)
    # cv2.imshow('frame', img)

    vd = CarDetector()
    bound_boxes = vd.detect_vehicles(img)
    img = get_image_with_box(img, bound_boxes)
    car_count = 0
    for object in bound_boxes:
        #id -> int
        #box -> tuple
        id, box = object
        x, y, w, h = box  
        #center is a tuple (x,y)
        box_center = find_center(x,y,w,h)
        
        if id == 2 or id == 7:
            color = (0,0,255)
            if is_point_inside(atm_polygon, box_center):
                car_count = car_count + 1
                color = (0,255,0)
            img = cv2.circle(img, box_center, radius=10, color = color, thickness=-1)
    print(f"filled : {car_count}\tvacant:{5-car_count}\ttotal : 5")
    cv2.imshow('dots', img)

    #randomly printing points on the image
    pt_array = np.random.randint(low=0, high=900, size=(100,2))
    for it in pt_array:
        x,y = it
        color = (255,0,0)
        if is_point_inside(atm_polygon, (x,y)):
            color =(0, 0, 255)
        #Draw a red circle with zero radius and -1 for filled circle
        img = cv2.circle(img, (x,y), radius=10, color = color, thickness=-1)
    cv2.imshow('fun',img)
    # iiii = is_point_inside((100,100))
    # print(iiii)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def count_cars_in_polygon(polygon_corners, img, slots = 5):

    vd = CarDetector()
    bound_boxes = vd.detect_vehicles(img)
    # img = get_image_with_box(img, bound_boxes)
    car_count = 0
    for object in bound_boxes:
        # id -> int          box -> tuple
        id, box = object
        x, y, w, h = box  
        #center is a tuple (x,y)
        box_center = find_center(x,y,w,h)
        
        if id == 2 or id == 7:
            # color = (0,0,255)
            if is_point_inside(polygon_corners, box_center):
                car_count = car_count + 1
                # color = (0,255,0)
            # img = cv2.circle(img, box_center, radius=10, color = color, thickness=-1)
    print(f"filled : {car_count}\tvacant:{slots-car_count}\ttotal : {slots}")
    # cv2.imshow('dots', img)
    return car_count


if __name__ == "__main__":
    # main("ppimages")
    # count_draw_cars("atm/atm_0img.png")

    b_time = time.perf_counter()

    image = cv2.imread("atm/atm_0img.png")
    _cars_filled = count_cars_in_polygon(atm_polygon, image)
    print(_cars_filled)

    e_time = time.perf_counter()

    print(e_time - b_time)
