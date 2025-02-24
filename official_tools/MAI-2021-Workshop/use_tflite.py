import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import onnxruntime

def preprocess_image(image_path, input_size=(1280, 720)):
    # 图像预处理（适配YOLOv5的预处理逻辑）
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img_array = np.array(img) / 255.0  # 归一化
    img_array = img_array[np.newaxis, ...].astype(np.float32)  # 添加batch维度
    return img_array

def predict_with_tflite():
    # 初始化TFLite解释器
    interpreter = tf.lite.Interpreter(model_path='F:\Projects\detection2025_realtime_camera_scence\official_tools\MAI-2021-Workshop\model_best.tflite')
    interpreter.allocate_tensors()

    # 获取输入输出张量信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 预处理图像
    input_data = preprocess_image(r'F:\Projects\datasets\RTCS_split\val\1_Portrait\12.jpg')

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

# ... 已有的tflite函数 ...

def predict_with_torch():
    """PyTorch模型推理函数"""
    # 模型加载
    model_path = r'F:\Projects\detection2025_realtime_camera_scence\official_tools\MAI-2021-Workshop\model_best.pt'
    model = torch.load(model_path)
    model.eval()
    
    # 预处理图像（保持与tflite相同处理）
    input_data = preprocess_image(r'F:\Projects\datasets\RTCS_split\val\1_Portrait\12.jpg')
    
    # 转换张量格式（NCHW）
    input_tensor = torch.from_numpy(input_data.transpose(0, 3, 1, 2))
    
    # 执行推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 后处理
    probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    print(f"Torch预测结果 - 类别: {predicted_class}, 置信度: {confidence:.4f}")

def predict_with_onnx():
    """ONNX模型推理函数"""
    # 创建推理会话
    onnx_path = r'F:\Projects\detection2025_realtime_camera_scence\official_tools\MAI-2021-Workshop\model_best.onnx'
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 预处理图像（添加NCHW转换）
    input_data = preprocess_image(r'F:\Projects\datasets\RTCS_split\val\1_Portrait\14.jpg', (640, 640))
    input_data = input_data.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    
    # 执行推理
    output = session.run([output_name], {input_name: input_data})
    
    # 后处理
    probabilities = tf.nn.softmax(output[0][0]).numpy()
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    print(f"ONNX预测结果 - 类别: {predicted_class}, 置信度: {confidence:.4f}")

if __name__ == "__main__":
    predict_with_tflite()
    # predict_with_torch()  # 新增调用
    predict_with_onnx()   # 新增调用






