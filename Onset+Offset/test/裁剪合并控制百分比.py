import cv2
import os
import dlib
import matplotlib.pyplot as plt

# 初始化 Dlib 的人脸检测器和地标预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\tengxunsp\CloFormer-main\model_else\shape_predictor_68_face_landmarks.dat")

# 设置默认坐标（手动输入的矩形框，描述眼睛和嘴巴区域）
default_coordinates = {
    "left_eye": (50, 60, 50, 30),  # 左眼区域 (x, y, width, height)
    "right_eye": (120, 60, 50, 30),  # 右眼区域 (x, y, width, height)
    "mouth": (75, 140, 80, 40)  # 嘴巴区域 (x, y, width, height)
}

def add_cropped_flow_to_base(vertex_image, flow_start_image, flow_end_image, eye_crop_size=80, mouth_expand_ratio=0.2,
                             weight_eye_mouth=0.4, smoothing_radius=15):
    # 转换为灰度图以供 Dlib 检测
    gray_vertex = cv2.cvtColor(vertex_image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray_vertex)
    if len(faces) == 0:
        print("No faces detected. Using default coordinates.")
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

    # 创建一个用于合成的光流图，初始为起始帧光流图
    combined_flow = flow_start_image.copy()

    # 只对裁剪区域进行加权合成
    # 合成眼睛区域
    eye_end_flow = flow_end_image[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3],
                   left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]
    eye_start_flow = combined_flow[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3],
                     left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]]

    # 计算平滑过渡区域（可以使用加权平均来平滑边缘）
    smoothed_eye_flow = smooth_transition(eye_start_flow, eye_end_flow, smoothing_radius, weight_eye_mouth)
    combined_flow[left_eye_bbox[1]:left_eye_bbox[1] + left_eye_bbox[3],
    left_eye_bbox[0]:left_eye_bbox[0] + left_eye_bbox[2]] = smoothed_eye_flow

    # 合成嘴巴区域
    mouth_end_flow = flow_end_image[mouth_bbox[1]:mouth_bbox[1] + mouth_bbox[3],
                     mouth_bbox[0]:mouth_bbox[0] + mouth_bbox[2]]
    mouth_start_flow = combined_flow[mouth_bbox[1]:mouth_bbox[1] + mouth_bbox[3],
                       mouth_bbox[0]:mouth_bbox[0] + mouth_bbox[2]]

    # 计算平滑过渡区域
    smoothed_mouth_flow = smooth_transition(mouth_start_flow, mouth_end_flow, smoothing_radius, weight_eye_mouth)
    combined_flow[mouth_bbox[1]:mouth_bbox[1] + mouth_bbox[3],
    mouth_bbox[0]:mouth_bbox[0] + mouth_bbox[2]] = smoothed_mouth_flow

    # 保证非裁剪区域保持为起始帧光流
    return combined_flow


def smooth_transition(start_image, end_image, radius, weight):
    """
    使用高斯模糊和加权平均进行平滑过渡。
    :param start_image: 起始图像（即起始帧光流）
    :param end_image: 结束图像（即结束帧光流）
    :param radius: 高斯模糊的半径
    :param weight: 叠加权重
    :return: 平滑过渡后的图像
    """
    # 计算权重
    weight_start = 1 - weight
    weight_end = weight

    # 使用高斯模糊平滑图像
    blurred_start = cv2.GaussianBlur(start_image, (radius, radius), 0)
    blurred_end = cv2.GaussianBlur(end_image, (radius, radius), 0)

    # 加权合成
    smoothed_image = cv2.addWeighted(blurred_start, weight_start, blurred_end, weight_end, 0)

    return smoothed_image


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


def process_batch(vertex_folder, flow_start_folder, flow_end_folder, output_folder, weight_eye_mouth=0.4):
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
        combined_flow = add_cropped_flow_to_base(vertex_image, flow_start_image, flow_end_image,
                                                 weight_eye_mouth=weight_eye_mouth)

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
process_batch(vertex_folder, flow_start_folder, flow_end_folder, output_folder, weight_eye_mouth=0.4)
