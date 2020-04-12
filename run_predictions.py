import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFilter
from perceptron2 import detect_red_light

def visualize(I, boxes, filename):
    new_image = ImageDraw.Draw(I)
    #boxes.append([0, 0, 50, 50])
    for x1, y1, x2, y2 in boxes:
        points = [(y1, x1), (y2, x1), \
        (y2, x2), (y1, x2), (y1, x1)]
        new_image.line(points, fill="green", width = 1)
    I.save(filename, "JPEG")

# set the path to the downloaded data:
data_path = './RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    img = I

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)
    visualize(img, preds[file_names[i]], "./imgs_3/test_{}.jpg".format(i + 1))

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_smooth.json'),'w') as f:
    json.dump(preds,f)
