import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFilter

# Extract an image of a traffic light
temp = Image.open("RedLights2011_Medium/RL-001.jpg")
temp = temp.crop((316, 154, 323, 171))
#temp = temp.filter(ImageFilter.BLUR)
# Normalize
k = np.asarray(temp).copy()
x, y, _ = k.shape
k = k.flatten()
k = k / np.linalg.norm(k)

# Extract different size image
temp = Image.open("RedLights2011_Medium/RL-010.jpg")
temp = temp.crop((321, 27, 349, 92))
#temp = temp.filter(ImageFilter.BLUR)

k2 = np.asarray(temp).copy()
x_2, y_2, _ = k2.shape
k2 = k2.flatten()
k2 = k2 / np.linalg.norm(k2)


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''


    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    '''
    BEGIN YOUR CODE
    '''
    row, col, _ = I.shape
    I = I.copy()
    threshold = 0.92

    results = np.zeros((row // 2, col - y))

    # Check for one of the traffic lights
    for i in range(row // 2):
        for j in range(col - y):
            region = I[i:i+x, j:j+y]
            region = region.flatten()
            norm = np.linalg.norm(region)
            if norm > 0:
                region = region / norm
            results[i, j] = np.inner(region, k)
    # Check for the other size traffic light
    results2 = np.zeros((row // 2, col - y_2))
    for i in range(row // 2):
        for j in range(col - y_2):
            region = I[i:i+x_2, j:j+y_2]
            region = region.flatten()
            norm = np.linalg.norm(region)
            if norm > 0:
                region = region / norm
            results2[i, j] = np.inner(region, k2)

    # Group into objects based on neighbors
    curr_label = 1
    filtered = results > threshold
    row, col = results.shape
    labels = np.zeros((row, col))
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            # If adjacent pixel in top or left is labeled use that label otherwise we place same label otherwise create new
            if filtered[i, j]:
                if i > 0:
                    if j > 0:
                        if labels[i - 1, j - 1] > 0:
                            labels[i, j] = labels[i - 1, j - 1]
                            continue
                    if labels[i - 1, j] > 0:
                        labels[i, j] = labels[i - 1, j]
                        continue
                    if j < col - 1:
                        if labels[i - 1, j + 1] > 0:
                            labels[i, j] = labels[i - 1, j + 1]
                            continue
                if j > 0:
                    if labels[i, j - 1] > 0:
                        labels[i, j] = labels[i, j - 1]
                        continue
                labels[i, j] = curr_label
                curr_label += 1
    # Create boxes for this class of traffic lights
    boxes = {}
    for i in range(row):
        for j in range(col):
            if labels[i, j] > 0:
                l = labels[i, j]
                if l in boxes:
                    y1, x1, y2, x2 = boxes[l]
                    boxes[l] = [min(i, y1), min(x1, j), \
                    max(y2, i), max(x2, j)]
                else:
                    boxes[l] = [i, j, i, j]
    for key in boxes:
        y1, x1, y2, x2 = boxes[key]
        boxes[key] = [y1, x1, y2 + x, x2 + y]

    # Group second labelings into groups
    curr_label = 1
    filtered = results2 > threshold
    row, col = results2.shape
    labels = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            # If adjacent pixel in top or left is labeled use that label otherwise we place same label otherwise create new
            if filtered[i, j]:
                if i > 0:
                    if j > 0:
                        if labels[i - 1, j - 1] > 0:
                            labels[i, j] = labels[i - 1, j - 1]
                            continue
                    if labels[i - 1, j] > 0:
                        labels[i, j] = labels[i - 1, j]
                        continue
                    if j < col - 1:
                        if labels[i - 1, j + 1] > 0:
                            labels[i, j] = labels[i - 1, j + 1]
                            continue
                if j > 0:
                    if labels[i, j - 1] > 0:
                        labels[i, j] = labels[i, j - 1]
                        continue
                labels[i, j] = curr_label
                curr_label += 1
    # Create boxes for these size traffic lights
    boxes2 = {}
    for i in range(row):
        for j in range(col):
            if labels[i, j] > 0:
                l = labels[i, j]
                if l in boxes2:
                    y1, x1, y2, x2 = boxes2[l]
                    boxes2[l] = [min(i, y1), min(x1, j), \
                    max(y2, i), max(x2, j)]
                else:
                    boxes2[l] = [i, j, i, j]

    for key in boxes2:
        y1, x1, y2, x2 = boxes2[key]
        boxes2[key] = [y1, x1, y2 + x_2, x2 + y_2]

    # Compile these boxes
    bounding_boxes = list(boxes.values())
    bounding_boxes.extend(list(boxes2.values()))
    '''
    END YOUR CODE
    '''
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes
