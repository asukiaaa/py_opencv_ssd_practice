# py_opencv_ssd_practice
SSD program with using opencv.

# Requirement
- python3
- [tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 1.9
- OpenCV 3.4

# Download model
```
mkdir data
cd data
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -xzvf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
# tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
wget https://raw.githubusercontent.com/opencv/opencv_extra/3.3.1/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt
cd ../
```

# Usage
```
# python3 ssd_mobilenet.py --pb "data/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb" --pbtxt "data/ssd_mobilenet_v1_coco.pbtxt"
python3 ssd_mobilenet.py --pb "data/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb" --pbtxt "data/ssd_mobilenet_v1_coco.pbtxt"
```

# References
- [TensorFlow Object Detection API で学習済みモデルを使って物体検出](http://robotics4society.com/2017/08/23/odapi_test/)
- [models/research/object_detection/g3doc/detection_model_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
