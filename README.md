# Workflow to retrain the coco neural net for detecting new set of objects

1) Install Ubuntu

2) Install Anaconda

3) The Tensorflow Object Detection API has been installed as documented in the installation instructions (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). This includes installing library dependencies, compiling the configuration protobufs and setting up the Python environment.

4) create environment with python version 2.7
```sh
conda create -n tf13 python=2.7
```

5) activate new environment
```sh
conda activate tf13
```
6) install tensorflow 1.3 in environment
for different tf versions or without gpu support choose another binary URL
https://www.tensorflow.org/install/install_linux
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl
(pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl)
```

7) check tensorflow installation with python
```sh
$ python
```
```sh
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

# install libraries (make sure you are still in env tf13)
## with apt-get
sudo apt-get install protobuf-compiler python-pil python-lxml

## with pip
sudo pip install jupyter
sudo pip install matplotlib

## alternatively you can install the dependencies (from the apt-get call) with pip 
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib

# clone tensorflow models repository.
git clone https://github.com/tensorflow/models.git ~/models

# checkout the commit, that works with tf1.3
# if you work with tf1.4 you need a later commit
cd ~/models
git checkout edcd29f2dbb4b3eaed387fe17cb5270f867aec42

# create a env variable for later use
export TF13MODELS=~/models

# store the variable in the environment
mkdir ~/anaconda3/envs/tf13/etc
mkdir ~/anaconda3/envs/tf13/etc/conda
mkdir ~/anaconda3/envs/tf13/etc/conda/activate.d
touch ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
echo export TF13MODELS=~/modelsmkdir >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh


# compile the protobuf libraries
# from ~/models/research/
cd ${TF13MODELS}/research
protoc object_detection/protos/*.proto --python_out=.

# if last call (protoc) is not installed, you need to install protobuf-compiler
## Make sure you grab the latest version
curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip

## Unzip
unzip protoc-3.2.0-linux-x86_64.zip -d protoc3

## Move protoc to /usr/local/bin/
sudo mv protoc3/bin/* /usr/local/bin/

## Move protoc3/include to /usr/local/include/
sudo mv protoc3/include/* /usr/local/include/

## Optional: change owner
sudo chwon [user] /usr/local/bin/protoc
sudo chwon -R [user] /usr/local/include/google

# add libraries to PYTHONPATH
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/tf13/lib/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:${TF13MODELS}/research:${TF13MODELS}/research/slim


## add to PYTHONPATH of tf13 env
echo export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/tf13/lib/python2.7/site-packages >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
echo 'export PYTHONPATH=$PYTHONPATH:${TF13MODELS}/research:${TF13MODELS}/research/slim' >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh

# testing installation
# You can test that you have correctly installed the Tensorflow Object Detection
# API by running the following command:
python ${TF13MODELS}/research/object_detection/builders/model_builder_test.py

# create a directory for data and nn
mkdir ~/bucket

# create a env variable for later use
export DATABUCKET=~/bucket
echo export DATABUCKET=~/bucket >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh

# create directory structure
mkdir ${DATABUCKET}/data


# get the pet dataset
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
mv images.tar.gz ${DATABUCKET}/data/.
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
mv annotations.tar.gz ${DATABUCKET}/data/.

# unpack data
cd ${DATABUCKET}/data
tar -xvf annotations.tar.gz
tar -xvf images.tar.gz

# convert the data into tensorflow record format
# (in tf1.4 the script path is ${TF14MODELS}/research/object_detection/dataset_tools/create_pet_tf_record.py
python ${TF13MODELS}/research/object_detection/create_pet_tf_record.py \
    --label_map_path=${TF13MODELS}/research/object_detection/data/pet_label_map.pbtxt \
    --data_dir=${DATABUCKET}/data \
    --output_dir=${DATABUCKET}/data


# add label map file of the pet data
cp ${TF13MODELS}/research/object_detection/data/pet_label_map.pbtxt ${DATABUCKET}/data/.

# download COCO-pretrained Model for Transfer Learning
mkdir ${DATABUCKET}/models
mkdir ${DATABUCKET}/models/model
cd ${DATABUCKET}/models/model
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
mv ${DATABUCKET}/models/model/faster_rcnn_resnet101_coco_11_06_2017/* ${DATABUCKET}/models/model/.

# Configure the Object Detection Pipeline
cp ${TF13MODELS}/research/object_detection/samples/configs/faster_rcnn_resnet101_pets.config ${DATABUCKET}/models/model/.
sed -i "s|PATH_TO_BE_CONFIGURED|"${DATABUCKET}"/data|g" ${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config

# creating directories for train and eval jobs
mkdir ${DATABUCKET}/models/model/train
mkdir ${DATABUCKET}/models/model/eval

# running the training job
python ${TF13MODELS}/research/object_detection/train.py \
   --logtostderr \
   --pipeline_config_path=${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config \
   --train_dir=${DATABUCKET}/models/model/train

# after some time run evaluation job concurrently
python ${TF13MODELS}/research/object_detection/eval.py \
   --logtostderr \
   --pipeline_config_path=${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config \
   --checkpoint_dir=${DATABUCKET}/models/model/train \
   --eval_dir=${DATABUCKET}/models/model/eval

# export the graph for inference
python ${TF13MODELS}/research/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config \
--trained_checkpoint_prefix ${DATABUCKET}/models/model/train/model.ckpt-7602 \
--output_directory=${DATABUCKET}/models/model/inference

# check detection quality with script



