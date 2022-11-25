# Yolov5-OpenCV-Cpp-Python-ROS

Adapted from [yolov5-opencv-cpp-python](https://github.com/doleron/yolov5-opencv-cpp-python)

Example of performing inference with ultralytics YOLO V5, OpenCV 4.5.4 DNN, C++ and Python

Futhermore, this code is adapted to receive ROS image messages and display the detections.

## Code explanation

+ Variable **net_path** defines the onnx network path. 
+ Variable **class_path** defines the class-file path.
+ The ros node listens on **/image** to get input images.
> NOTE: The code depends on the output dimension of your network model, which means the variables **dimensions and rows** in ros.cpp should be the exact same size of output dimensions.


## Prerequisites

Make sure you have already on your system:

- Any modern Linux OS (tested on Ubuntu 20.04)
- OpenCV 4.5.4+
- Python 3.7+ (only if you are intended to run the python program)
- GCC 9.0+ (only if you are intended to run the C++ program)

**IMPORTANT**!!! Note that OpenCV versions prior to 4.5.4 will not work at all.

## Running the ROS/C++ program

The C++ code is [here](cpp/yolo.cpp).

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS/Yolo_ROS
mkdir build && cd build
cmake ../
make
./yolo_ros
```


## Running the python script

The python code is [here](python/yolo.py).

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS
python python/yolo.py 
```

If your machine/OpenCV install are CUDA capable you can try out running using the GPU:

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS
python python/yolo.py cuda
python python/yolo-tiny.py cuda
```

## Running the C++ program

The C++ code is [here](cpp/yolo.cpp).

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS/cpp
mkdir build && cd build
cmake ../
make
./yolo_example
```



## Which YOLO version should I use?

This repository uses YOLO V5 but it is not the only YOLO version out there. You can read [this article](https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109) to learn more about YOLO versions and choose the more suitable one for you.

## Exporting yolo v5 models to .onnx format

Check here: https://github.com/ultralytics/yolov5/issues/251

My commands were:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
And then to convert the model:

```bash
$ python3 export.py --weights yolov5n.pt --img 640 --include onnx
export: data=data/coco128.yaml, weights=['yolov5n.pt'], imgsz=[640], batch_size=1, device=cpu, half=False, inplace=False, train=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v6.0-192-g436ffc4 torch 1.10.1+cu102 CPU

Fusing layers... 
Model Summary: 213 layers, 1867405 parameters, 0 gradients

PyTorch: starting from yolov5n.pt (4.0 MB)

ONNX: starting export with onnx 1.10.2...
/home/user/workspace/smartcam/yolov5/models/yolo.py:57: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
ONNX: export success, saved as yolov5n.onnx (7.9 MB)

Export complete (1.33s)
Results saved to /home/doleron/workspace/smartcam/yolov5
Visualize with https://netron.app
Detect with `python detect.py --weights yolov5n.onnx` or `model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.onnx')
Validate with `python val.py --weights yolov5n.onnx`
$ 
```
