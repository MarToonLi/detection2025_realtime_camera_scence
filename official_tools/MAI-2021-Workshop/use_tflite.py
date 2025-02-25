import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import onnxruntime
import os
import cv2
from augmentations import classify_transforms



def preprocess_image(image_path, input_size=(224, 224), fp16=False):
    im0 = cv2.imread(image_path)
    print("像素和：", im0.sum())
    
    im = classify_transforms(input_size[0])(im0)  # transforms
    im = im[None]                      # 添加batch维度
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    im = np.array(im)
    print("像素和：", im.sum())
    print("output shape: ", im.shape)
    
    return im


def preprocess_image_beifen(image_path, input_size=(224, 224)):
    # 图像预处理（适配YOLOv5的预处理逻辑）
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img_array = np.array(img)  # 归一化
    # img_array = np.array(img) / 255.0  # 归一化
    img_array = img_array[np.newaxis, ...].astype(np.float32)  # 添加batch维度
    img_array = img_array.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    return img_array


def predict_with_tflite(tflite_model_path):
    # 初始化TFLite解释器
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # 获取输入输出张量信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 预处理图像
    input_data = preprocess_image(r'F:\Projects\datasets\RTCS_split\val\1_Portrait\12.jpg', (224, 224))

    # 执行推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 获取输出结果
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # 后处理（假设是分类输出）
    probabilities = tf.nn.softmax(output_data[0]).numpy()
    print(probabilities.shape)
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

def predict_with_onnx(onnx_model_path):
    """ONNX模型推理函数"""
    # 创建推理会话
    onnx_model_path
    session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 预处理图像（添加NCHW转换）
    input_data = preprocess_image(r'F:\Projects\datasets\RTCS_split\val\1_Portrait\12.jpg', (224, 224))
    
    # 执行推理
    output = session.run([output_name], {input_name: input_data})

    
    meta = session.get_modelmeta().custom_metadata_map  # metadata
    if 'stride' in meta:
        stride, names = int(meta['stride']), eval(meta['names'])
    
    # 后处理
    probabilities = tf.nn.softmax(output[0][0]).numpy()  
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    print(f"ONNX预测结果 - 类别: {predicted_class}, 类别名称: {names[predicted_class]}, 置信度: {confidence:.4f}")


def predict_with_torch(path):
    """PyTorch模型推理函数"""
    # 加载PyTorch模型
    model = torch.load(path, map_location='cpu')
    model.eval()
    model.to('cpu')

    # 预处理图像（保持与ONNX一致的预处理）
    input_data = preprocess_image(r'F:\Projects\datasets\RTCS_split\val\1_Portrait\14.jpg', (224, 224))
    input_tensor = torch.from_numpy(input_data.transpose(0, 3, 1, 2))  # NHWC -> NCHW

    # 执行推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理（适配分类输出）
    probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    print(f"PyTorch预测结果 - 类别: {predicted_class}, 置信度: {confidence:.4f}")



if __name__ == "__main__":
    
    ROOT = r"F:\Projects\detection2025_realtime_camera_scence\official_tools\MAI-2021-Workshop\in_output"
    test_name = "test1"
    onnx_model_name = "best"
    
    
    ROOT = os.path.join(ROOT, test_name)
    pt_model_path = os.path.join(ROOT, "model_{}.pt".format(onnx_model_name))
    onnx_model_path = os.path.join(ROOT, "model_{}.onnx".format(onnx_model_name))
    tflite_model_path = os.path.join(ROOT, "model_{}.tflite".format(onnx_model_name))
    
    pt_model_path = r"F:\Projects\detection2025_realtime_camera_scence\yolov5\runs\train-cls\exp5\weights\last.pt"
    print("pt_model_path: ", pt_model_path)
    
    predict_with_onnx(onnx_model_path)   # 新增调用
    predict_with_tflite(tflite_model_path)







