from veho_det import CarDetector
import cv2
from pathlib import Path

#this file contains the code to draw bounding boxes around the objects 
#mentioned in the dictonary below

# the new image is saved either in the location specified by save_location 
# or in the same folder after renaming it as "boxed_{original name}" 
def draw_boxes(raw_img_path, save_location = None):

    #creating the file name to image with boxes
    #using Path
    _file_path = Path(raw_img_path)
    directory = _file_path.parent
    extension = _file_path.suffix
    file_name = _file_path.stem
    
    if save_location == None:
        save_location = f"{directory}/boxed_{file_name}{extension}"

    print(f"{raw_img_path}\n{save_location}\n")

    img = cv2.imread(raw_img_path)

    #resize to just fit within screen (no other reason)
    height = 900
    width = int(1920/1080 * height)
    img = cv2.resize(img, (width, height))

    cv2.imshow('original image',img)

    dict_id_object = {
        1:"bicycle",
        2:"car",
        3:"motorcycle",
        5:"bus",
        7:"truck"}

    vd = CarDetector()
    bound_boxes = vd.detect_vehicles(img)

    for object in bound_boxes:
        #id -> int
        #box -> tuple
        id, box = object
        x, y, w, h = box

        # if not id in dict_id_object:
            # continue     
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
        cv2.putText(img, dict_id_object[id], (x,y), 0, .4, (100, 200, 0), 1)
        cv2.putText(img, "vehicles:" + str(len(bound_boxes)), (20,30), 0, 1, (100, 100, 0), 2)
        cv2.imshow("after detection" , img)
                
    cv2.imwrite(save_location, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    draw_boxes("/home/melvin/Downloads/objDetect_2022/ppimages/20220624_152847.jpg")