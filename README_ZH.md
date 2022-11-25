# Yolov5-OpenCV-Cpp-Python-ROS

使用OpenCV 4.5.4推理YOLOv5模型，分别使用C++，Python和ROS实现。


**基于[yolov5-opencv-cpp-python](https://github.com/doleron/yolov5-opencv-cpp-python)修改。**
在原代码的基础上使用CMake编译（能够更方便的定义程序路径），并且加入了对ROS传入图片的支持。

## 代码解释
+ 变量**net_path**定义了onnx网络模型路径.
+ 变量 **class_path**定义了分类文件的路径.
+ ROS节点订阅**/image**话题来获取输入图像.
> NOTE: The code depends on the output dimension of your network model, which means the variables **dimensions and rows** in ros.cpp should be the exact same size of output dimensions.
> 注意: 网络推理的计算取决于网络模型的输出维度, 也就是说ros.cpp中的变量**dimensions and rows**应该与其一致.

## 环境配置
- 任何Linux OS (在Ubuntu 18.04上测试)
- OpenCV 4.5.4+
- Python 3.7+(可选)
- GCC 9.0+(可选)
- ROS melodic(可选)

**注意**!!! 先于4.5.4的OpenCV版本不会正常运行。

## 使用ROS/C++推理

C++/ROS代码在[Yolo_ROS/ros.cpp](Yolo_ROS/ros.cpp)。

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS/Yolo_ROS
mkdir build && cd build
cmake ../
make
./yolo_ros
```


## 使用python推理

Python代码在[python/yolo.py](python/yolo.py).

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS
python python/yolo.py 
```

使用GPU运行：

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS
python python/yolo.py cuda
python python/yolo-tiny.py cuda
```

## 使用C++推理

C++代码在[cpp/yolo.cpp](cpp/yolo.cpp).

```bash
git clone https://github.com/YellowAndGreen/Yolov5-OpenCV-Cpp-Python-ROS.git
cd Yolov5-OpenCV-Cpp-Python-ROS/cpp
mkdir build && cd build
cmake ../
make
./yolo_example
```

## 导出Yolov5 模型到onnx格式

https://github.com/ultralytics/yolov5/issues/251

我的指令是:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```
然后转换模型:

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
