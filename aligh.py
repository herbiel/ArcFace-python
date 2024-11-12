import face_alignment
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 初始化人脸对齐模型
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')  # 使用 'cuda' 以利用 GPU

# 加载图像
image_path = '67500106d17a28145f59abe0a05e1c3b.jpeg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 将图像从 BGR 转换为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检测人脸关键点
preds = fa.get_landmarks(image_rgb)

# 定义对齐函数
def align_face(image, landmarks):
    # 关键点索引，通常使用眼睛和嘴巴进行对齐
    left_eye = landmarks[36:42]  # 左眼
    right_eye = landmarks[42:48]  # 右眼

    # 计算眼睛中心
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)

    # 计算旋转角度
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    # 计算旋转矩阵
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # 旋转图像
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return aligned_image

# 可视化对齐结果
def aligh_image(image):
    if preds is not None:
        for face in preds:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aligned_image = align_face(image_rgb, face)
    return aligned_image