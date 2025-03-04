#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python
@File    ：test_aligh.py
@Author  ：herbiel
@Date    ：2024/11/12 16:46
'''
import face_alignment
import cv2
import numpy as np

# 初始化人脸对齐模型
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

# 加载图像
image_path = "asserts/e60d70f679c81a112b1d2ca0bcfa992b.jpeg"  # 替换为你的图像路径
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检测人脸关键点
preds = fa.get_landmarks(image_rgb)

# 定义对齐函数
def align_face(image, landmarks):
    left_eye = landmarks[36:42]  # 左眼
    right_eye = landmarks[42:48]  # 右眼
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned_image

# 可视化对齐结果
if preds is not None:
    for face in preds:
        aligned_image = align_face(image_rgb, face)
        cv2.imshow("Aligned Face", aligned_image)
        cv2.waitKey(0)
else:
    print("未检测到人脸！")
