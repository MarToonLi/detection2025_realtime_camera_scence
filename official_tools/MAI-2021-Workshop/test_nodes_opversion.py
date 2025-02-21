import onnx

def check_opset_nodes(model_path):
    # 加载ONNX模型
    model = onnx.load(model_path)
    
    # 获取所有opset导入信息
    opset_imports = {opset.domain: opset.version for opset in model.opset_import}
    
    # 遍历计算图节点
    for node in model.graph.node:
        
        if node.name == "LayerNormalization":
        
            # 获取节点域（默认ai.onnx）
            print(node.domain)
            domain = node.domain or 'ai.onnx'
            
            # 检查对应的opset版本
            if opset_imports.get(domain, 0) == 13:
                print(f"Node '{node.name}' uses opset version 13 (domain: {domain})")

import onnx
from onnx import helper, numpy_helper

def replace_unsqueeze_with_reshape(model_path, output_path):
    model = onnx.load(model_path)
    
    # 创建新节点列表
    new_nodes = []
    
    for node in model.graph.node:
        if node.op_type == 'Unsqueeze':
            # 获取维度参数（适配不同ONNX版本）
            axes = None
            if node.attribute:  # opset < 13
                axes = [a.ints[0] for a in node.attribute if a.name == 'axes'][0]
            else:  # opset >= 13 通过输入获取
                axes = numpy_helper.to_array(model.graph.initializer[node.input[1]]).tolist()
            
            # 生成形状常量节点
            shape_name = node.name + "_new_shape"
            new_shape = [1 if i in axes else 0 for i in range(len(model.graph.value_info))]  # 简化示例
            shape_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[shape_name],
                value=helper.make_tensor(
                    name=shape_name + '_value',
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(new_shape)],
                    vals=new_shape
                )
            )
            
            # 创建替换的Reshape节点
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[node.input[0], shape_name],
                outputs=node.output,
                name=node.name + '_reshape'
            )
            
            new_nodes.extend([shape_node, reshape_node])
        else:
            new_nodes.append(node)
    
    # 更新模型结构
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    onnx.save(model, output_path)




if __name__ == "__main__":
    model_path = "/ns_data/projets/detection2025_realtime_camera_scence/official_tools/MAI-2021-Workshop/model_CLIP.onnx"  # 修改为实际模型路径
    # check_opset_nodes(model_path)
    
    
    replace_unsqueeze_with_reshape(
    model_path,
    "modified_model.onnx"
)
