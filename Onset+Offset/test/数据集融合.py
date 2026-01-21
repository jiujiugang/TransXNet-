import os
import shutil

def merge_datasets_with_suffix(original_dir, fused_dir, output_dir, suffix="_fused"):
    """
    将原始数据集和融合后的数据集合并到同一文件夹，并为融合后的数据集添加后缀区分。

    Args:
        original_dir (str): 原始数据集文件夹路径。
        fused_dir (str): 融合后的数据集文件夹路径。
        output_dir (str): 合并后的输出文件夹路径。
        suffix (str): 融合后的数据文件名后缀（默认为 "_fused"）。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理原始数据集
    for filename in os.listdir(original_dir):
        original_path = os.path.join(original_dir, filename)
        if os.path.isfile(original_path):
            output_path = os.path.join(output_dir, filename)
            shutil.copy2(original_path, output_path)  # 保留原始文件名
            print(f"Copied original: {filename} -> {output_path}")

    # 处理融合后的数据集
    for filename in os.listdir(fused_dir):
        fused_path = os.path.join(fused_dir, filename)
        if os.path.isfile(fused_path):
            # 添加后缀到文件名
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}{suffix}{ext}"
            output_path = os.path.join(output_dir, new_filename)
            shutil.copy2(fused_path, output_path)  # 添加后缀后保存
            print(f"Copied fused: {filename} -> {output_path}")

# 示例用法
original_dataset_dir = r"D:\tengxunsp\CloFormer-main\dataset\data\flow"  # 原始数据集路径
fused_dataset_dir = r"D:\tengxunsp\CloFormer-main\TVL3"       # 融合后数据集路径
output_dataset_dir = r"D:\tengxunsp\CloFormer-main\更改后数据集-平滑\fuse_PH"     # 合并后的数据集输出路径

merge_datasets_with_suffix(original_dataset_dir, fused_dataset_dir, output_dataset_dir, suffix="_fused")
