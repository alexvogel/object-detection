import sys
import os
import argparse
import re

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageFont

import time
from scipy.stats import norm


def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, scores, classes, classname, thickness_box=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)

    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        #class_id = int(classes[i])
        color = None
        if (classname[i] == "Green"):
            color = (0, 255, 0)
        elif (classname[i] == "Yellow"):
            color = (255, 255, 0)
        elif (classname[i] == "Red"):
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)

        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness_box, fill=color)


#        draw.text((0, 0), str(int(scores[i] * 100)) + ": " + classname[i] ,(255,255,255), font=font)

        draw.text((left, bot), str(int(scores[i] * 100)) + "%: " + classname[i] ,color, font=font)

def detect(image_path, output_dir, id_name, image_tensor, detection_boxes, detection_scores, detection_classes):
    
    image = Image.open(image_path)
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

    with tf.Session(graph=detection_graph) as sess:                
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                        feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.5
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        tmp_output_file_path = output_dir + "/" + os.path.basename(image_path)
        filepath, ext = os.path.splitext(tmp_output_file_path)
        output_file_path = filepath + ".png"
        print("extension is " + ext)

        # Each class with be represented by a differently colored box

        # create list of classnames for this image
        classnames = []

        # log results
        print("====")
        for i in range(0, len(scores)):
            classnames.append(id_name[str(int(classes[i]))])
            print(str(scores[i]) + ": " + str(int(classes[i])) + " -> " + id_name[str(int(classes[i]))])

        print("----")

        print(str(len(classes)) + " classe(s) detected in " + output_file_path)

        print("====")

        # draw boxes
        draw_boxes(image, box_coords, scores, classes, classnames)

        fig = plt.figure(figsize=(12, 8))

#        plt.figtext(x, y, )

#        plt.imshow(image)

        #plt.imsave(fname=output_file_path, arr=image)
        plt.imsave(fname=output_file_path, arr=image, format="png")

        # close window to free resources
        image.close()


VERSION = "1.0"
DATE = "20170-12-27"

cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

parser = argparse.ArgumentParser(description='a tool for detect objects in an image',
                             epilog='author: alexander.vogel@prozesskraft.de | version: ' + VERSION + ' | date: ' + DATE)
parser.add_argument("--frozen_graph", metavar='PATH', action='store', type=str, required=True, default="", help="frozen tensorflow graph file (*.pb)")
parser.add_argument("--input_image", metavar='PATH', action='store', type=str, required=True, default="", help="image file or directory with images to perform object detection on")
parser.add_argument("--output_dir", metavar='PATH', action='store', type=str, required=False, default="outdir", help="directory for the detected images. for debugging purposes")
parser.add_argument("--label_map", metavar='PATH', action='store', type=str, required=True, default="", help="label_map_file.pbtxt")
args = parser.parse_args()

error = 0

if not os.path.isfile(args.frozen_graph):
    print("error: file does not exist: " + args.frozen_graph)
    error += 1
if not os.path.exists(args.input_image):
    print("error: file or directory does not exist: " + args.input_image)
    error += 1
if not os.path.isfile(args.label_map):
    print("error: file does not exist: " + args.label_map)
    error += 1
if os.path.exists(args.output_dir):
    print("error: output directory already exists. delete or rename.", args.output_dir)
    error += 1

if(error > 0):
	sys.exit(1)

images_path = []

# if its a single file
if os.path.isfile(args.input_image):
    images_path.append(args.input_image)

# if its a directory
else:
    filenames_in_dir = os.listdir(args.input_image)

    # put together dir and filename
    for filename in filenames_in_dir:
        images_path.append(args.input_image + "/" + filename)

# print all paths
#for lol in images_path:
#    print(lol)

# read and parse label_map into a dict
id_name = {}
print("reading label map " + args.label_map)
with open(args.label_map) as f:
    lines = f.read().splitlines()

    label_id = None
    label_name = None

    for line in lines:
        # match for label id
        match_id = re.match("^  id: (\d+)$", line)
        if(match_id):
            label_id = match_id.group(1)
#            print("ID matched: " + label_id)

        # match for label name
        match_name = re.match("^  name: '(\w+)'$", line)
        if(match_name): 
            label_name = match_name.group(1)
#            print("NAME matched: " + label_name)

            # add to dict
            id_name[label_id] = label_name

            # reset
            label_id = None
            label_name = None

# print label map
for key in id_name.keys():
    print(key + " -> " + id_name[key])

# creating the output directory
os.makedirs(args.output_dir)

print("start reading frozen graph file " + args.frozen_graph)
detection_graph = load_graph(args.frozen_graph)
print("end reading frozen graph file")

print("start extract tensors as interface")

# The input placeholder for the image.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects. 
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

print("end extract tensors as interface")

for image_path in images_path:
    detect(image_path, args.output_dir, id_name, image_tensor, detection_boxes, detection_scores, detection_classes)
