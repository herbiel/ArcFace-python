#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：check_face.py
@Author  ：herbiel
@Date    ：2024/11/15 10:53 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：check_face.py
@Author  ：herbiel
@Date    ：2024/11/15 10:53 
'''
import cv2
import dlib
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)

def rotate_image(image, angle):
    """旋转图像到指定角度"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def detect_faces_dlib(image):
    """使用 Dlib 检测人脸"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image)
    return faces

def load_image(image_source):
    """加载图像，可以是 URL 或本地路径"""
    try:
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source)
            response.raise_for_status()  # 检查请求是否成功
            image = Image.open(BytesIO(response.content))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(image_source)

        if image is None:
            raise ValueError("Error: Image not found or unable to read.")
        return image
    except Exception as e:
        logging.error(f"Failed to load image from {image_source}: {e}")
        return None

def find_faces_by_rotation(image_source):
    """顺时针旋转图像直到检测到人脸"""
    image = load_image(image_source)
    if image is None:
        return None

    # 尝试不同的旋转角度
    for angle in range(0, 360, 90):  # 每次旋转90度
        rotated_image = rotate_image(image, angle)
        faces = detect_faces_dlib(rotated_image)

        if faces:
            logging.info(f"Detected {len(faces)} face(s) at angle {angle} degrees.")
            # 显示检测到的人脸
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(rotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)  # 返回检测到人脸的图像

    logging.warning("No faces detected after rotating through all angles.")
    return None  # 如果没有检测到人脸，返回 None

# 调用函数，传入图像的 URL 或本地路径
# img = find_faces_by_rotation('https://monas-001.oss-ap-southeast-5.aliyuncs.com/image/025166536f5138cc493e71736ece4da1.jpeg')
# if img is not None:
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
# else:
#     print("No faces detected in the image.")
