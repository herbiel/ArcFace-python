#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：main.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
from fastapi import FastAPI,Body,HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image

from arcface_detect import getfacesim
import cv2
import face_alignment
from skimage import io
import numpy as np
# Create a FastAPI instance
app = FastAPI()

@app.on_event("startup")
async def startup_event():
     global fa
     # 初始化人脸对齐方法
     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu',face_detector='blazeface')


def face_align_V2(img):
    input_image = io.imread(img)  # 替换为你的图像路径

    # 获取人脸关键点
    preds = fa.get_landmarks(input_image)

    # 检查是否检测到人脸
    if preds is not None and len(preds) > 0:
        # 取第一个检测到的人脸
        landmarks = preds[0]

        # 计算眼睛的坐标
        left_eye = landmarks[36:42].mean(axis=0)  # 左眼的关键点
        right_eye = landmarks[42:48].mean(axis=0)  # 右眼的关键点

        # 计算眼睛的角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))  # 计算角度

        # 旋转图像
        (h, w) = input_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(input_image, M, (w, h))
        return cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)


def read_image_from_url(url):
    # 这里是读取图像的函数，具体实现根据你的需求来
    # 例如，可以使用 requests 库来获取图像并使用 cv2 读取

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def align_face(image, landmarks):
    left_eye = landmarks[36:42]  # 左眼关键点
    right_eye = landmarks[42:48]  # 右眼关键点
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned_image

# Define a Pydantic model for the request body



# Define a POST endpoint
@app.post("/api/predict/facesmi")
def post_facesim(
        image1: str = Body(embed=True,alias="image1", min_length=10),
        image2: str = Body(embed=True,alias="image2", min_length=10),
):
    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    try:
        # 检测第一张图中的人脸
        img1_ori = read_image_from_url(image1)
        img2_ori = read_image_from_url(image2)
        result = getfacesim(img1_ori, img2_ori)
        print(f"{image1} and {image2} sim is {result}")
        return {
            "code": 200,
            "error": None,
            "score": str(result)
        }
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
