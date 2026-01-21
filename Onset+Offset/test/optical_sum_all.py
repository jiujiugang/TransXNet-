import cv2
import os
import dlib
import matplotlib.pyplot as plt

# 初始化 Dlib 的人脸检测器和地标预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\tengxunsp\CloFormer-main\model_else\shape_predictor_68_face_landmarks.dat")

# 设置默认坐标（手动输入的矩形框，描述眼睛和嘴巴区域）
default_coordinates = {
    "left_eye": (30, 40, 80, 80),  # 左眼区域 (x, y, width, height)
    "right_eye": (60, 40, 80, 80),  # 右眼区域 (x, y, width, height)
    "mouth": (50, 60, 100, 60)  # 嘴巴区域 (x, y, width, height)
}


def add_cropped_flow_to_base(vertex_image, flow_start_image, flow_end_image, eye_crop_size=80, mouth_expand_ratio=0.2):
    # 转换为灰度图以供 Dlib 检测
    gray_vertex = cv2.cvtColor(vertex_image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray_vertex)
    if len(faces) == 0:
        print("No faces detected. Using default coordinates.")

        # 使用默认的矩形框坐标来描述眼睛和嘴巴区域
        left_eye_bbox = default_coordinates["left_eye"]
        right_eye_bbox = default_coordinates["right_eye"]
        mouth_bbox = default_coordinates["mouth"]
    else:
        # 获取第一个人脸
        face = faces[0]
        landmarks = predictor(gray_vertex, face)

        # 提取眼睛和嘴巴的坐标
        left_eye = [landmarks.part(i) for i in range(36, 42)]  # 左眼
        right_eye = [landmarks.part(i) for i in range(42, 48)]  # 右眼
        mouth = [landmarks.part(i) for i in range(48, 68)]  # 嘴巴

        # 计算眼睛区域的矩形框（x, y, width, height）
        left_eye_bbox = get_bbox(left_eye, eye_crop_size)
        right_eye_bbox = get_bbox(right_eye, eye_crop_size)

        # 计算嘴巴区域的矩形框（x, y, width, height）
        mouth_bbox = get_bbox(mouth, expand=mouth_expand_ratio)

    # 裁剪并合成光流
    combined_flow = flow_start_image.copy()

    # 裁剪眼睛区域并合成
    eye_end_flow = flow_end_image[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3],
                   left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]
    combined_flow[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3],
    left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]] += eye_end_flow

    # 裁剪嘴巴区域并合成
    mouth_end_flow = flow_end_image[mouth_bbox[1]:mouth_bbox[1] + mouth_bbox[3],
                     mouth_bbox[0]:mouth_bbox[0] + mouth_bbox[2]]
    combined_flow[mouth_bbox[1]:mouth_bbox[1] + mouth_bbox[3],
    mouth_bbox[0]:mouth_bbox[0] + mouth_bbox[2]] += mouth_end_flow

    return combined_flow


def get_bbox(points, expand=0):
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)

    # 添加扩展
    width = max_x - min_x
    height = max_y - min_y
    min_x = max(0, min_x - int(width * expand))
    max_x = max_x + int(width * expand)
    min_y = max(0, min_y - int(height * expand))
    max_y = max_y + int(height * expand)

    return (min_x, min_y, max_x - min_x, max_y - min_y)  # 返回 (x, y, width, height)


def process_batch(vertex_folder, flow_start_folder, flow_end_folder, output_folder):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0  # 初始化计数器

    # 遍历起始帧光流文件夹中的所有图像
    for file_name in os.listdir(flow_start_folder):
        vertex_path = os.path.join(vertex_folder, file_name)
        flow_start_path = os.path.join(flow_start_folder, file_name)
        flow_end_path = os.path.join(flow_end_folder, file_name)

        # 确保对应的文件存在
        if not os.path.exists(vertex_path) or not os.path.exists(flow_end_path):
            print(f"Skipping {file_name}: Missing vertex or flow end file.")
            continue

        # 读取图像
        vertex_image = cv2.imread(vertex_path)
        flow_start_image = cv2.imread(flow_start_path)
        flow_end_image = cv2.imread(flow_end_path)

        # 执行裁剪和合成操作
        combined_flow = add_cropped_flow_to_base(vertex_image, flow_start_image, flow_end_image)

        # 保存结果
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, combined_flow)
        print(f"Processed and saved: {output_path}")

        processed_count += 1  # 计数加1

    print(f"Total processed images: {processed_count}")  # 打印总计数


# 示例文件夹路径
vertex_folder = r"D:\tengxunsp\CloFormer-main\dataset\data\apex"  # 顶点帧文件夹
flow_start_folder = r"D:\tengxunsp\CloFormer-main\dataset\data\flow"  # 起始帧光流文件夹
flow_end_folder = r"D:\tengxunsp\CloFormer-main\dataset\dataoff\of-ap-all"  # 结束帧光流文件夹
output_folder = r"D:\tengxunsp\CloFormer-main\TVL1\ap-of-all"  # 输出目录

# 批量处理
process_batch(vertex_folder, flow_start_folder, flow_end_folder, output_folder)
