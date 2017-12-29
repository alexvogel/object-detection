# from https://github.com/swirlingsand/deeper-traffic-lights.git
# put it in the tensorflow/models/research/object_detection

import sys
import tensorflow as tf
import yaml
import os
import argparse
from object_detection.utils import dataset_util
import cv2
import numpy as np
import matplotlib.pyplot as plt


#flags = tf.app.flags
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#FLAGS = flags.FLAGS

DEBUG = True
DEBUGDIR = "debug_dir"

LABEL_DICT =  {
    "Green" : 1,
    "Yellow" : 2,
    "Red" : 3,
    }

def draw_boxes_and_write(image_path, boxes, classname, thickness_box=4):
    """Draw bounding boxes on the image"""

    image = Image.open(image_path)

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)

    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness_box, fill=color)


#        draw.text((0, 0), str(int(scores[i] * 100)) + ": " + classname[i] ,(255,255,255), font=font)

        draw.text((left, bot), str(int(scores[i] * 100)) + "%: " + classname[i] ,color, font=font)

    output_file_path = DEBUGDIR + "/" + image_path.split("/")[-1]
    print("output debug " + output_file_path)
    #plt.imsave(fname=output_file_path, arr=image, format="png")

    image.close()


def create_tf_example(example):
    
    # Bosch
    #height = 720 # Image height
    #width = 1280 # Image width

    # print size of image
    img = cv2.imread(example['path'])
    print("shape: " + str(img.shape) + " " + example['path'])

    height = img.shape[0]
    width = img.shape[1]

    filename = example['path'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        if(os.path.isfile(filename)):
            encoded_image = fid.read()
        else:
            print("ERROR: File does not exist: ", filename)
            return None


    image_format = 'png'.encode() 

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    # for debugging
    debug_boxes = []

    for box in example['annotations']:
        #if box['occluded'] is False:
        #print("adding box")
        xmin_unnormalized = float(box['xmin'])
        xmax_unnormalized = float(box['xmin'] + box['x_width'])
        ymin_unnormalized = float(box['ymin'])
        ymax_unnormalized = float(box['ymin'] + box['y_height'])

        xmins.append(float(xmin_unnormalized / width))
        xmaxs.append(float(xmax_unnormalized / width))
        ymins.append(float(ymin_unnormalized / height))
        ymaxs.append(float(ymax_unnormalized / height))
        classes_text.append(box['class'].encode())
        classes.append(int(LABEL_DICT[box['class']]))

        debug_boxes.append([xmin_unnormalized, xmax_unnormalized, ymin_unnormalized, ymax_unnormalized])

    # for debugging
    if(DEBUG):
    	draw_boxes_and_write(example['path'], debug_boxes, classes_text)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_yaml", type=str, default="", help="yaml file of bosch traffic lights database")
    parser.add_argument("--output_record", type=str, default="", help="output tensorflow record file")
    args = parser.parse_args()

    writer = tf.python_io.TFRecordWriter(args.output_record)
    
    # BOSCH
    INPUT_YAML = args.input_yaml
    examples = yaml.load(open(INPUT_YAML, 'rb').read())

    #examples = examples[:10]  # for testing
    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")

    for i in range(len(examples)):
        examples[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), examples[i]['filename']))
    
    if(DEBUG):
    	print("creating debug directory")
    	makedir(args.debug_dir)

    counter = 0
    for example in examples:
        tf_example = create_tf_example(example)

        if(tf_example != None):
        	writer.write(tf_example.SerializeToString())
        else:
            print("Skipped last entry")
            next

        if counter % 10 == 0:
            print("Percent done", (counter/len_examples)*100)
        counter += 1

    writer.close()



if __name__ == '__main__':
	tf.app.run()
