# detection2025_realtime_camera_scence
https://codalab.lisn.upsaclay.fr/competitions/21563#participate-get_data

## 未知概念
将模型转换为 TFLite 格式

训练后量化   Post-Training Quantization

量化感知训练   Quantization Aware Training


## 构建过程
- [x] 构建github项目：detection_2025_realtime_camera_scence
    - [x] source源文件
    - [x] 竞赛相关工具源码
    - [x] 日志和checkpoints
    - [x] requirements.txt
- [x] 数据集整理：
- [x] 模型选择yolov5
- [x] FP32-torch模型转换为tflite模型脚本
- [x] INT8量化-torch模型转换为tflite模型脚本


## 尝试记录
- ❌ torch模型->onnx模型->tflite模型的方式不可行：它依赖于onnx2tf开源库，但是tensorflow支持的opset的范围仅限于14~18。
经常会遇到一个问题：BackendIsNotSupposedToImplementIt: Unsqueeze version 13 is not implemented.该问题从14年持续到现在无法解决。
模型测试是基于CLIP模型。它表明这种转换方式不稳定，影响模型的改造。
https://github.com/onnx/onnx-tensorflow
https://github.com/onnx/tensorflow-onnx
- ✅ yolov5: 提供了多种模型转换方式，包括FP32->tflite、INT8->tflite。
网友实现：https://blog.csdn.net/qq_40214464/article/details/122582080
鉴于此，该流程可以推进。
- 模型确定为yolov5剪枝版模型
- 数据集如何组织？
- 多GPU训练





## 常用命令
单卡运行模式：
python classify/train.py --model yolov5s-cls.pt  --epochs 10 --img 224  --batch-size 64  --data /ns_data/datasets/RTCS

多卡运行模式：
python -m torch.distributed.run --nproc_per_node 2 --master_port 1 classify/train.py --model yolov5s-cls.pt --data /ns_data/datasets/RTCS --epochs 10 --img 224 --device 0,1

## 数据集组织形式
### 图像分类任务：
root
├── data
│   ├── RTCS
│   │   ├── train
│   │   │   ├── 0
│   │   │   ├── 1
│   │   │   ├── 2
|   |   |── val
│   │   │   ├── 0
│   │   │   ├── 1
│   │   │   ├── 2
|   |   |── test
│   │   │   ├── 0
│   │   │   ├── 1
│   │   │   ├── 2
注：
1. 调用时，只需要将data参数修改为 /root/data/RTCS
2. 训练时，如果不存在test文件夹，将把val文件夹作为验证文件夹
3. workers如果设置过大，会报错 ConnectionResetError: [Errno 104] Connection reset by peer

### 图像检测任务

### 图像分割任务







## 环境配置
1. 目前安装tensorflow和torch的较新版本，都需要cuda至少达到11.0版本，因此本机显卡能够支持的cuda的版本需要大于11.0





## 模型模块构建
1. 如果涉及到不同深度学习框架之间的模型转换，且是通过onnx中间格式转换，torch代码中尽量避免使用unsqueeze，因为tensorflow支持的onnx版本范围是14~18
