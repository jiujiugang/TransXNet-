import cv2
import numpy as np
import dlib
import random
import matplotlib.pyplot as plt

# 载入 Dlib 人脸检测器和地标预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\tengxunsp\CloFormer-main\model_else\shape_predictor_68_face_landmarks.dat")  # 替换为你自己的路径

def mask_non_eye_mouth_regions_fixed(flow_image, vertex_image, num_masks=20):
    # 将顶点帧图像转换为灰度图像，以便 Dlib 进行人脸检测
    gray_vertex = cv2.cvtColor(vertex_image, cv2.COLOR_BGR2GRAY)

    # 使用 Dlib 检测人脸
    faces = detector(gray_vertex)

    if len(faces) == 0:
        print("No faces detected in vertex frame.")
        return None

    # 获取第一个检测到的人脸
    face = faces[0]

    # 使用 Dlib 获取人脸地标
    landmarks = predictor(gray_vertex, face)

    # 提取眼睛和嘴巴的地标点
    left_eye = [landmarks.part(i) for i in range(36, 42)]  # 左眼
    right_eye = [landmarks.part(i) for i in range(42, 48)]  # 右眼
    mouth = [landmarks.part(i) for i in range(48, 68)]  # 嘴巴

    # 获取区域边界框（添加 10% 的边界扩展，避免区域过小）
    def get_bbox(points, expand=0.1):
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        width = max_x - min_x
        height = max_y - min_y

        # 添加扩展
        min_x = max(0, min_x - int(width * expand))
        max_x = min(flow_image.shape[1], max_x + int(width * expand))
        min_y = max(0, min_y - int(height * expand))
        max_y = min(flow_image.shape[0], max_y + int(height * expand))

        return min_x, min_y, max_x, max_y

    # 计算眼睛和嘴巴的边界框
    left_eye_bbox = get_bbox(left_eye)
    right_eye_bbox = get_bbox(right_eye)
    mouth_bbox = get_bbox(mouth)

    # 划分光流图为 14×14 的小块
    flow_image_h, flow_image_w = flow_image.shape[:2]
    block_size = flow_image_h // 14  # 每块的大小 (16×16)
    mask_grid = np.zeros((14, 14), dtype=bool)  # 用于记录被掩码的块

    # 将眼睛和嘴巴区域转为块索引范围
    def bbox_to_block_indices(bbox):
        min_row, min_col = bbox[1] // block_size, bbox[0] // block_size
        max_row, max_col = bbox[3] // block_size, bbox[2] // block_size
        return min_row, max_row, min_col, max_col

    # 获取眼睛和嘴巴区域的块索引
    left_eye_blocks = bbox_to_block_indices(left_eye_bbox)
    right_eye_blocks = bbox_to_block_indices(right_eye_bbox)
    mouth_blocks = bbox_to_block_indices(mouth_bbox)

    # 将眼睛和嘴巴的块标记为 True，表示不可掩码
    for min_row, max_row, min_col, max_col in [left_eye_blocks, right_eye_blocks, mouth_blocks]:
        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                if 0 <= i < 14 and 0 <= j < 14:  # 防止越界
                    mask_grid[i, j] = True

    # 随机掩码非眼睛和嘴巴区域的 20 个块
    maskable_indices = [(i, j) for i in range(14) for j in range(14) if not mask_grid[i, j]]
    selected_indices = random.sample(maskable_indices, min(num_masks, len(maskable_indices)))

    # 创建掩码图像
    masked_image = flow_image.copy()
    for i, j in selected_indices:
        y1, y2 = i * block_size, (i + 1) * block_size
        x1, x2 = j * block_size, (j + 1) * block_size
        masked_image[y1:y2, x1:x2] = 0  # 将块置为 0

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Optical Flow Image")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Masked Optical Flow Image")
    ax2.axis('off')

    plt.show()

    return masked_image


# 示例使用
vertex_image_path = r'/dataset/data/apex/006_006_1_2.jpg'  # 顶点帧图像路径
flow_image_path = r'/dataset/data/flow/006_006_1_2.jpg'  # 光流图像路径

# 读取顶点帧和光流图
vertex_image = cv2.imread(vertex_image_path)
flow_image = cv2.imread(flow_image_path)

# 执行掩码操作
masked_flow_image = mask_non_eye_mouth_regions_fixed(flow_image, vertex_image)
