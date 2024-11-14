#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：rotate_image.py
@Author  ：herbiel
@Date    ：2024/11/13 18:48 
'''
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO


def fetch_and_rotate_image(url):
    # 从 URL 获取图像
    response = requests.get(url)

    # 打开图像
    img = Image.open(BytesIO(response.content))

    # 旋转图像 90 度
    rotated_image = img.rotate(-90, expand=True)  # 顺时针旋转

    # 将 PIL 图像转换为 NumPy 数组
    image_array = np.array(rotated_image)

    # 将 RGB 转换为 BGR（OpenCV 使用 BGR 格式）
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return bgr_image

