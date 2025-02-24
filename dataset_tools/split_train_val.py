import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    按4:1比例分割数据集
    :param input_dir: 原始数据集路径（包含带类别子文件夹的training目录）
    :param output_dir: 输出根目录（将自动创建train/val子目录）
    :param train_ratio: 训练集比例（默认0.8即4:1）
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 遍历所有类别文件夹
    for class_folder in tqdm(os.listdir(os.path.join(input_dir, 'training'))):
        if not os.path.isdir(os.path.join(input_dir, 'training', class_folder)):
            continue
            
        # 获取类别所有图片
        class_path = os.path.join(input_dir, 'training', class_folder)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 按比例分割
        train_files, val_files = train_test_split(
            images, 
            train_size=train_ratio, 
            shuffle=True, 
            stratify=None, 
            random_state=42
        )
        
        # 创建类别子目录并复制文件
        for split_name, split_files in [('train', train_files), ('val', val_files)]:
            dest_dir = os.path.join(output_dir, split_name, class_folder)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in split_files:
                shutil.copy(
                    os.path.join(class_path, f),
                    os.path.join(dest_dir, f)
                )

if __name__ == "__main__":
    # 使用示例（根据实际路径修改）
    split_dataset(
        input_dir=r"F:\Projects\datasets\RTCS",  # 原始数据集路径
        output_dir=r"F:\Projects\datasets\RTCS_split"  # 输出路径
    )
