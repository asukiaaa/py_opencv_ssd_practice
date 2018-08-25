# WIP
This readme is not completed.
I'm happy if giving me some advice.

# Executed commands

Put this repository in ~/gitprojects/tensorflow
```
mkdir -p ~/gitprojects/tensorflow
cd ~/gitprojects/tensorflow
git clone git@github.com:asukiaaa/py_opencv_ssd_practice.git
```

Install tensorflow with referencing a [readme](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

```
sudo apt install git python3-pip protobuf-compiler python3-pil python3-lxml python3-tk
sudo pip3 install tensorflow Cython contextlib2 jupyter matplotlib
```

Download data and execute program with referencing a [readme](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md).

```
cd ~/gitprojects/tensorflow
git clone https://github.com/tensorflow/models.git
cd models/research
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

Install cocoapi for python3.
```
sudo pip3 install pycocotools
# cd ~/gitprojects
# git clone https://github.com/cocodataset/cocoapi.git
# cd cocoapi/PythonAPI
# make
# cp -r pycocotools ~/gitprojects/tensorflow/models/research/
```

Configure path of object detection for python.
```
cd ~/gitprojects/tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Create record.
```
cd ~/gitprojects/tensorflow/models/research/
python3 object_detection/dataset_tools/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

Readme introduces a way to use GCS(Google Cloud Storage) but I don't want to use that.

Run locally with referencing a [readme](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md).

Edit `~/gitprojects/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config`.

Remove `fine_tune_checkpoint` and change `from_detection_checkpoint`.
```
  # fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
  # from_detection_checkpoint: true
  from_detection_checkpoint: false
```

Edit `input_path` and `label_map_path` in train_input_reader and eval_input_reader.
```
  tf_record_input_reader {
    # input_path: "PATH_TO_BE_CONFIGURED/pet_faces_val.record-?????-of-00010"
    input_path: "../pet_faces_val.record-?????-of-00010"
  }
  # label_map_path: "PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt"
  label_map_path: "data/pet_label_map.pbtxt"
```

Set all flags as false in batch_norm. ref: https://github.com/opencv/opencv/issues/11570
```
        batch_norm {
          train: false,
          scale: true,
          center: true,
          decay: 0.9997,
          epsilon: 0.001,
        }
```

Add list for category_index.values on line about 390 in `~/gitprojects/tensorflow/models/research/object_detection/model_lib.py`. ref: https://github.com/tensorflow/models/issues/4780
```
      eval_metric_ops = eval_util.get_eval_metric_ops_for_evaluators(
          eval_config,
          list(category_index.values()),
          eval_dict)
```

After editing ssd_mobilenet_v1_pets.config, execute model_main.py to create models.

```
cd ~/gitprojects/tensorflow/models/research/object_detection
PIPELINE_CONFIG_PATH="`pwd`/samples/configs/ssd_mobilenet_v1_pets.config"
MODEL_DIR="`pwd`/model_dir"
# NUM_TRAIN_STEPS=50000
# NUM_EVAL_STEPS=2000

NUM_TRAIN_STEPS=10
NUM_EVAL_STEPS=10
# 253 seconds

start=`date +%s`
python3 model_main.py \
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
# WIP
# # Change this line on `research/object_detection/models/embedded_ssd_mobilenet_v1_feature_extractor.py` in model repository.
# -          mobilenet_v1.mobilenet_v1_arg_scope(is_training=None)):
# +          mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
```

Export graph. ref: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md

```
cd ~/gitprojects/tensorflow/models/research/object_detection/model_dir
CHECKPOINT_NUMBER=10
python3 ../export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ../samples/configs/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory exported_graphs
#     --pipeline_config_path pipeline.config \
```

Export graph.pbtxt from frozen graph.
```
cd ~/gitprojects/tensorflow/models/research/object_detection/model_dir/exported_graphs
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_ssd.py
python3 tf_text_graph_ssd.py \
  --input frozen_inference_graph.pb
  --output graph.pbtxt \
  --num_classes \
  --min_scale \
  --max_scale \
  --num_layers \
  --aspect_ratios
```

```
# cd ~/gitprojects/tensorflow/models/research/object_detection
# CHECKPOINT_NUMBER=10
# python3 models/embedded_ssd_mobilenet_v1_feature_extractor.py \
#     --input_type image_tensor \
#     --pipeline_config_path samples/configs/ssd_mobilenet_v1_pets.config \
#     --trained_checkpoint_prefix model_dir/model.ckpt-${CHECKPOINT_NUMBER} \
#     --output_directory exported_graphs
```

```
# ??? how to run
# cd ~/gitprojects/tensorflow/models/research/object_detection/model_dir
# CHECKPOINT_NUMBER=10
# python3 ../models/embedded_ssd_mobilenet_v1_feature_extractor.py \
#     --input_type image_tensor \
#     --pipeline_config_path pipeline.config \
#     --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
#     --output_directory exported_graphs
```

```
# # Need optimization?
# cd ~/gitprojects/tensorflow
# git clone https://github.com/tensorflow/tensorflow.git code
# cd ~/gitprojects/tensorflow/models/research/object_detection/model_dir/exported_graphs
# python3 ~/gitprojects/tensorflow/code/tensorflow/python/tools/optimize_for_inference.py \
#   --input frozen_inference_graph.pb \
#   --output opt_graph.pb \
#   --input_names image_tensor \
#   --output_names "num_detections,detection_scores,detection_boxes,detection_classes" \
#   --placeholder_type_enum 4 \
#   --frozen_graph
# # ref:
# # http://answers.opencv.org/question/175699/readnetfromtensorflow-fails-on-retrained-nn/
# # https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
```


Use exported data.
```
cd ~/gitprojects/tensorflow/py_opencv_ssd_practice/
OBJECT_DETECTION_DIR=~/gitprojects/tensorflow/models/research/object_detection
# cp ${OBJECT_DETECTION_DIR}/data/pet_label_map.pbtxt ./
cp ${OBJECT_DETECTION_DIR}/model_dir/graph.pbtxt ./
cp ${OBJECT_DETECTION_DIR}/exported_graphs/frozen_inference_graph.pb ./
# cp ${OBJECT_DETECTION_DIR}/exported_graphs/opt_graph.pb ./
cp ${OBJECT_DETECTION_DIR}/data/pet_label_map.pbtxt ./
cp ${OBJECT_DETECTION_DIR}/data/mscoco_label_map.pbtxt ./
# cp ${OBJECT_DETECTION_DIR}/model_dir/exported_graphs/saved_model/saved_model.pb ./
# python3 ssd_mobilenet.py --pb "frozen_inference_graph.pb" --pbtxt "graph.pbtxt" # reading caffe??
python3 ssd_mobilenet.py --pb "frozen_inference_graph.pb"
```

Load config or pbtxt. ref: https://github.com/tensorflow/models/issues/4450

```
DONE (t=1.87s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.004
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.005
WARNING:tensorflow:num_readers has been reduced to 10 to match input file shards.
強制終了
asuki@asuki-ThinkPad-S1-Yoga:~/gitprojects/tensorflow/models/research/object_detection$ end=`date +%s`
asuki@asuki-ThinkPad-S1-Yoga:~/gitprojects/tensorflow/models/research/object_detection$ runtime=$((end-start))
asuki@asuki-ThinkPad-S1-Yoga:~/gitprojects/tensorflow/models/research/object_detection$ echo $runtime seconds
5669 seconds
```

Clear data.
```
rm -r ~/gitprojects/tensorflow/models/research/object_detection/model_dir
```

# References
- https://github.com/tensorflow/models/blob/master/research/object_detection/README.md
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md
