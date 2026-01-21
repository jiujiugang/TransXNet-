import os
import shutil
import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(r'C:\Users\24291\Desktop\CloFormer-main\dataset\data\442.xlsx')

# 确保 'flowimg' 列的数据是字符串类型，并处理缺失值
df['flowimg'] = df['flowimg'].fillna('').astype(str)

# 确保 'subject' 列的数据是字符串类型，并处理缺失值
df['subject'] = df['subject'].fillna('').astype(str)

# 确保 'Filename' 列的数据是字符串类型，并处理缺失值
df['filename'] = df['filename'].fillna('').astype(str)

# 指定原始图片文件夹路径
original_images_dir = r'/dataset/data/apex'

# 指定新数据集的文件夹路径
base_dir = r'/dataset/data/apex_loso'

# 遍历每个唯一的 subject
for subject in df['subject'].unique():
    if not subject:  # 跳过空的 subject
        continue

    # 确保 subject 为三位数字格式
    formatted_subject = subject.zfill(3)

    subject_df = df[df['subject'] == subject]

    # 为每个 subject 创建主文件夹
    subject_dir = os.path.join(base_dir, formatted_subject)
    os.makedirs(subject_dir, exist_ok=True)

    # 创建 u_test 和 u_train 文件夹
    u_test_dir = os.path.join(subject_dir, 'u_test')
    os.makedirs(u_test_dir, exist_ok=True)
    u_train_dir = os.path.join(subject_dir, 'u_train')
    os.makedirs(u_train_dir, exist_ok=True)

    # 在 u_test 中创建 label 文件夹
    for label in subject_df['label'].unique():
        os.makedirs(os.path.join(u_test_dir, str(label)), exist_ok=True)

    # 处理每个 subject 的数据
    for _, row in subject_df.iterrows():
        img_file = row['flowimg']
        if not img_file:  # 跳过空的 img 文件名
            continue

        label = row['label']
        # 直接在 apex 文件夹中查找图片
        src_file = os.path.join(original_images_dir, img_file)  # 直接使用 img_file

        # 移动到 u_test 对应的 label 文件夹
        dst_dir = os.path.join(u_test_dir, str(label))
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, img_file)

        # 处理文件是否存在
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)  # 使用 shutil.move 如果需要移动文件而非复制
        else:
            print(f"文件 {src_file} 不存在")

    # 处理 u_train 文件夹中的数据
    other_subjects_df = df[df['subject'] != subject]

    for label in other_subjects_df['label'].unique():
        os.makedirs(os.path.join(u_train_dir, str(label)), exist_ok=True)

    for _, row in other_subjects_df.iterrows():
        img_file = row['flowimg']
        if not img_file:  # 跳过空的 img 文件名
            continue

        label = row['label']
        # 直接在 apex 文件夹中查找图片
        src_file = os.path.join(original_images_dir, img_file)  # 直接使用 img_file

        # 移动到 u_train 对应的 label 文件夹
        dst_dir = os.path.join(u_train_dir, str(label))
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, img_file)

        # 处理文件是否存在
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)  # 使用 shutil.move 如果需要移动文件而非复制
        else:
            print(f"文件 {src_file} 不存在")
