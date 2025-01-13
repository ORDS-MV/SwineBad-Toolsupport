import json
import os
import cv2

path_GT = '...' # path to the GT with the table annotations 
path_JSON = os.path.join(path_GT,'JSON')
path_images = os.path.join(path_GT,'images')

# select specific file
year = '...'
file_name = '...'

json_file = os.path.join(path_JSON,year,file_name+'.json')

#load json
with open(json_file, "r") as jsonFile:
    data = json.load(jsonFile)

# load image
image_name = os.path.join(path_images,year,file_name+'.jpg')
img = cv2.imread(image_name)

color_list = [(0,255,0),(0,0,255),(255,0,0)]


for idx,table in enumerate(data.get('shapes')):
    points = table.get('points')
    xy_tuple1 = (int(points[0][0]),int(points[0][1]))
    xy_tuple2 = (int(points[1][0]), int(points[1][1]))
    cv2.rectangle(img,xy_tuple1, xy_tuple2, color_list[idx], 3)

from PIL import Image
Image.fromarray(img).show()
input("Press Enter to exit...")











