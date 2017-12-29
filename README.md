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
```

7) check tensorflow installation with python
```sh
python
```
```sh
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

8) install libraries (make sure you are still in env tf13)
8.1) With apt-get
```sh
sudo apt-get install protobuf-compiler python-pil python-lxml
```

8.2) With pip
```sh
sudo pip install jupyter
sudo pip install matplotlib
```

8.3) Alternatively you can install the dependencies (from the apt-get call) with pip 
```sh
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

9) Clone tensorflow models repository.
```sh
git clone https://github.com/tensorflow/models.git ~/models
```

10) Checkout the commit, that works with tf1.3 (if you work with tf1.4 or a later version you need a later commit)
```sh
cd ~/models
git checkout edcd29f2dbb4b3eaed387fe17cb5270f867aec42
```

11) Create a env variable for later use
```sh
export TF13MODELS=~/models
```

12) Store the variable in the environment
```sh
mkdir ~/anaconda3/envs/tf13/etc
mkdir ~/anaconda3/envs/tf13/etc/conda
mkdir ~/anaconda3/envs/tf13/etc/conda/activate.d
touch ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
echo export TF13MODELS=~/modelsmkdir >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
```

13) Compile the protobuf libraries (from ~/models/research/)
```sh
cd ${TF13MODELS}/research
protoc object_detection/protos/*.proto --python_out=.
```

14) Add libraries to PYTHONPATH
```sh
export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/tf13/lib/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:${TF13MODELS}/research:${TF13MODELS}/research/slim
```

15) Add libraries to PYTHONPATH in environment tf13
```sh
echo export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/tf13/lib/python2.7/site-packages >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
echo 'export PYTHONPATH=$PYTHONPATH:${TF13MODELS}/research:${TF13MODELS}/research/slim' >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
```

16) You can test that you have correctly installed the Tensorflow Object Detection API by running the following command:
```sh
python ${TF13MODELS}/research/object_detection/builders/model_builder_test.py
```

17) create a working directory for data and neural net
```sh
mkdir ~/bucket
```

18) create a env variable for later use
```sh
export DATABUCKET=~/bucket
echo export DATABUCKET=~/bucket >> ~/anaconda3/envs/tf13/etc/conda/activate.d/env_vars.sh
```

19) create directory structure
```sh
mkdir ${DATABUCKET}/data
```

20) get a dataset with new objects - in this example the pet database
```sh
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
mv images.tar.gz ${DATABUCKET}/data/.
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
mv annotations.tar.gz ${DATABUCKET}/data/.
```

21) unpack data
```sh
cd ${DATABUCKET}/data
tar -xvf annotations.tar.gz
tar -xvf images.tar.gz
```

22) convert the data into tensorflow record format using the dataset tool from the tensorflow models repo.
```sh
python ${TF13MODELS}/research/object_detection/create_pet_tf_record.py \
    --label_map_path=${TF13MODELS}/research/object_detection/data/pet_label_map.pbtxt \
    --data_dir=${DATABUCKET}/data \
    --output_dir=${DATABUCKET}/data
```

23) add the label map file of the pet data
```sh
cp ${TF13MODELS}/research/object_detection/data/pet_label_map.pbtxt ${DATABUCKET}/data/.
```

24) download COCO-pretrained Model for Transfer Learning
```sh
mkdir ${DATABUCKET}/models
mkdir ${DATABUCKET}/models/model
cd ${DATABUCKET}/models/model
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
mv ${DATABUCKET}/models/model/faster_rcnn_resnet101_coco_11_06_2017/* ${DATABUCKET}/models/model/.
```

25) Configure the Object Detection Pipeline
```sh
cp ${TF13MODELS}/research/object_detection/samples/configs/faster_rcnn_resnet101_pets.config ${DATABUCKET}/models/model/.
sed -i "s|PATH_TO_BE_CONFIGURED|"${DATABUCKET}"/data|g" ${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config
```

26) Create directories for train and eval jobs
```sh
mkdir ${DATABUCKET}/models/model/train
mkdir ${DATABUCKET}/models/model/eval
```

27) Run the training job
```sh
python ${TF13MODELS}/research/object_detection/train.py \
   --logtostderr \
   --pipeline_config_path=${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config \
   --train_dir=${DATABUCKET}/models/model/train
```

28) After the first ckpt-file appears in the train directory the evaluation job can launched concurrently
```sh
python ${TF13MODELS}/research/object_detection/eval.py \
   --logtostderr \
   --pipeline_config_path=${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config \
   --checkpoint_dir=${DATABUCKET}/models/model/train \
   --eval_dir=${DATABUCKET}/models/model/eval
```

29) Export the trained variables into a frozen graph for inference
```sh
python ${TF13MODELS}/research/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ${DATABUCKET}/models/model/faster_rcnn_resnet101_pets.config \
--trained_checkpoint_prefix ${DATABUCKET}/models/model/train/model.ckpt-7602 \
--output_directory=${DATABUCKET}/models/model/inference
```

30) Run the detection with script and check the quality with the generated images in output_dir
```sh
python scripts/detect_obj.py \
  --frozen_graph ${DATABUCKET}/models/model/inference/frozen_inference_graph.pb \
  --label_map ${DATABUCKET}/data/pet_label_map.pbtxt \
  --input_image ${DATABUCKET}/data/images \
  --output_dir outdir
```


