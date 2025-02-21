import os
import json
from tqdm import tqdm
from PIL import Image

def convert_to_coco(root_path, output_file):
    # 初始化数据结构
    coco = {
        "images": [],
        "categories": [],
        "annotations": []
    }
    
    # 生成类别映射
    class_folders = sorted(os.listdir(root_path))
    for idx, folder in enumerate(class_folders):
        # 解析文件夹名称格式 "数字_类别名称"
        folder_id, folder_name = folder.split('_', 1)
        coco["categories"].append({
            "id": int(folder_id),
            "name": folder_name,
            "supercategory": "none"
        })
    
    # 遍历所有图像
    ann_id = 1
    for cat in coco["categories"]:
        folder_path = os.path.join(root_path, f"{cat['id']:02d}_{cat['name']}")
        for img_name in tqdm(os.listdir(folder_path), desc=cat['name']):
            img_path = os.path.join(folder_path, img_name)
            
            # 获取图像尺寸
            with Image.open(img_path) as img:
                width, height = img.size
                
            # 添加图像记录
            img_id = len(coco["images"]) + 1
            coco["images"].append({
                "id": img_id,
                "file_name": os.path.join(f"{cat['id']:02d}_{cat['name']}", img_name),
                "width": width,
                "height": height
            })
            
            # 添加标注记录
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat["id"]
            })
            ann_id += 1
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)

if __name__ == "__main__":
    convert_to_coco(
        root_path="data/path",  # 修改为实际路径
        output_file="coco_classification.json"
    )
