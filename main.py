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

from arcface_detect import getfacesim
import cv2
import numpy as np
#import face_alignment
# Create a FastAPI instance
app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     global fa
#     # 初始化人脸对齐方法
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

def read_image_from_url(url):
    # 这里是读取图像的函数，具体实现根据你的需求来
    # 例如，可以使用 requests 库来获取图像并使用 cv2 读取
    import requests
    from io import BytesIO
    from PIL import Image

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
class Item(BaseModel):
    image1: str
    image2: str


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
        # preds1 = fa.get_landmarks(img1_ori)
        # if preds1 is not None:
        #     # 如果检测到人脸，进行对齐
        #     aligned_img1 = align_face(img1_ori, preds1[0])
        # else:
        #     print("第一张图未检测到人脸！")
        #     return None

        # # 检测第二张图中的人脸
        # preds2 = fa.get_landmarks(img2_ori)
        # if preds2 is not None:
        #     # 如果检测到人脸，进行对齐
        #     aligned_img2 = align_face(img2_ori, preds2[0])
        # else:
        #     print("第二张图未检测到人脸！")
        #     return None
        result = getfacesim(img1_ori, img2_ori)
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
