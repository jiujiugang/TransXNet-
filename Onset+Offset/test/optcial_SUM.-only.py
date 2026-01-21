import cv2
import os
import dlib
import matplotlib.pyplot as plt

# 初始化 Dlib 的人脸检测器和地标预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\tengxunsp\CloFormer-main\model_else\shape_predictor_68_face_landmarks.dat")


def add_cropped_flow_to_base(vertex_image, flow_start_image, flow_end_image, output_dir, eye_crop_size=80, mouth_expand_ratio=0.2):
    # 转换为灰度图以供 Dlib 检测
    gray_vertex = cv2.cvtColor(vertex_image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray_vertex)
    if len(faces) == 0:
        print("No faces detected.")
        return None

    # 获取第一个人脸
    face = faces[0]
    landmarks = predictor(gray_vertex, face)

    # 提取眼睛和嘴巴的坐标
    left_eye = [landmarks.part(i) for i in range(36, 42)]  # 左眼
    right_eye = [landmarks.part(i) for i in range(42, 48)]  # 右眼
    mouth = [landmarks.part(i) for i in range(48, 68)]  # 嘴巴

    # 获取中心点坐标
    def get_center(points):
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        return center_x, center_y

    # 获取边界框坐标
    def get_fixed_bbox(center, size):
        half_size = size // 2
        min_x = max(0, center[0] - half_size)
        max_x = center[0] + half_size
        min_y = max(0, center[1] - half_size)
        max_y = center[1] + half_size
        return min_x, min_y, max_x, max_y

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
        return min_x, min_y, max_x, max_y

    # 眼睛区域
    left_eye_center = get_center(left_eye)
    right_eye_center = get_center(right_eye)

    left_eye_bbox = get_fixed_bbox(left_eye_center, eye_crop_size)
    right_eye_bbox = get_fixed_bbox(right_eye_center, eye_crop_size)

    # 合并左右眼区域
    eye_region = (
        min(left_eye_bbox[1], right_eye_bbox[1]),  # min_y
        max(left_eye_bbox[3], right_eye_bbox[3]),  # max_y
        min(left_eye_bbox[0], right_eye_bbox[0]),  # min_x
        max(left_eye_bbox[2], right_eye_bbox[2])   # max_x
    )

    # 嘴巴区域
    mouth_bbox = get_bbox(mouth, expand=mouth_expand_ratio)

    # 裁剪结束帧与顶点帧光流的眼睛和嘴巴区域
    eye_end_flow = flow_end_image[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]]
    mouth_end_flow = flow_end_image[mouth_bbox[1]:mouth_bbox[3], mouth_bbox[0]:mouth_bbox[2]]

    # 将裁剪的光流区域加回到起始帧与顶点帧光流
    combined_flow = flow_start_image.copy()

    # 加入眼睛区域
    combined_flow[eye_region[0]:eye_region[1], eye_region[2]:eye_region[3]] += eye_end_flow

    # 加入嘴巴区域
    combined_flow[mouth_bbox[1]:mouth_bbox[3], mouth_bbox[0]:mouth_bbox[2]] += mouth_end_flow

    # 生成输出路径
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "combined_flow.jpg")
    cv2.imwrite(output_file_path, combined_flow)

    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(flow_start_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Flow Start")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(combined_flow, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Combined Flow")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return output_file_path


# 示例使用
vertex_image_path = r"D:\tengxunsp\CloFormer-main\dataset\data\apex\006_006_1_2.jpg"  # 顶点帧图像路径
flow_start_image_path = r"D:\tengxunsp\CloFormer-main\dataset\data\flow\006_006_1_2.jpg"  # 起始帧光流
flow_end_image_path = r"D:\tengxunsp\CloFormer-main\dataset\dataoff\flow_442(of-ap)-classic\006\006_1_2\006_006_1_2.jpg"  # 结束帧光流

output_dir = r"D:\tengxunsp\CloFormer-main\TVL1\OUTPUT"  # 输出目录

vertex_image = cv2.imread(vertex_image_path)
flow_start_image = cv2.imread(flow_start_image_path)
flow_end_image = cv2.imread(flow_end_image_path)

# 裁剪并加回到起始帧光流
output_file = add_cropped_flow_to_base(vertex_image, flow_start_image, flow_end_image, output_dir)
