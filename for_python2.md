```
sudo apt install git python-pip protobuf-compiler python-pil python-lxml python-tk
sudo pip install tensorflow Cython contextlib2 jupyter matplotlib pycocotools
```

```
mkdir -p ~/gitprojects/tensorflow
cd ~/gitprojects/tensorflow
git clone https://github.com/tensorflow/models.git
cd models/research
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

Setup path for python.
```
cd ~/gitprojects/tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

```
cd ~/gitprojects/tensorflow/models/research/
python object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

```
cd ~/gitprojects/tensorflow/models/research/object_detection
PIPELINE_CONFIG_PATH="`pwd`/samples/configs/ssd_mobilenet_v1_pets.config"
MODEL_DIR="`pwd`/model_dir"
# NUM_TRAIN_STEPS=50000
# NUM_EVAL_STEPS=2000

NUM_TRAIN_STEPS=10
NUM_EVAL_STEPS=10
# 259 seconds

start=`date +%s`
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
end=`date +%s`

runtime=$((end-start))
echo $runtime seconds
```

```
cd ~/gitprojects/tensorflow/models/research/object_detection/model_dir
CHECKPOINT_NUMBER=10
python ../export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path pipeline.config \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory exported_graphs
```

```
cd ~/gitprojects/tensorflow/tensorflow-ssd/
OBJECT_DETECTION_DIR=~/gitprojects/tensorflow/models/research/object_detection
# cp ${OBJECT_DETECTION_DIR}/data/pet_label_map.pbtxt ./
cp ${OBJECT_DETECTION_DIR}/model_dir/graph.pbtxt ./
cp ${OBJECT_DETECTION_DIR}/model_dir/exported_graphs/frozen_inference_graph.pb ./
cp ${OBJECT_DETECTION_DIR}/model_dir/exported_graphs/opt_graph.pb ./
cp ${OBJECT_DETECTION_DIR}/data/pet_label_map.pbtxt ./
cp ${OBJECT_DETECTION_DIR}/data/mscoco_label_map.pbtxt ./
# cp ${OBJECT_DETECTION_DIR}/model_dir/exported_graphs/saved_model/saved_model.pb ./
python ssd_mobilenet.py
# cv2.error: OpenCV(3.4.2) /io/opencv/modules/dnn/src/tensorflow/tf_importer.cpp:495: error: (-2:Unspecified error) Input layer not found: Preprocessor/map/while/NextIteration in function 'connect'
```

Clear data.
```
rm -r ~/gitprojects/tensorflow/models/research/object_detection/model_dir
```
