#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：main.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
from fastapi import FastAPI,Body,HTTPException
import requests
from io import BytesIO
from PIL import Image
import httpx
#from arcface_detect import getfacesim
import cv2
import numpy as np
app = FastAPI()

from  arcface.engine import *

from config import  APPID,SDKKey

@app.on_event("startup")
async def startup_event():
    # 激活接口,首次需联网激活
    global face_engine,mask
    res = ASFOnlineActivation(APPID, SDKKey)
    if (MOK != res and MERR_ASF_ALREADY_ACTIVATED != res):
        print("ASFActivation fail: {}".format(res))
    else:
        print("ASFActivation sucess: {}".format(res))

    # 获取激活文件信息
    res, activeFileInfo = ASFGetActiveFileInfo()

    if (res != MOK):
        print("ASFGetActiveFileInfo fail: {}".format(res))
    else:
        print(f"{activeFileInfo}")

    # 获取人脸识别引擎
    face_engine = ArcFace()

    # 需要引擎开启的功能
    mask = ASF_FACE_DETECT | ASF_FACERECOGNITION

    # 初始化接口
    res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 30, 10, mask)
    if (res != MOK):
        print("ASFInitEngine fail: {}".format(res))
    else:
        print("ASFInitEngine sucess: {}".format(res))


async def read_image_from_url(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Define a Pydantic model for the request body

def getfacesim(img1, img2):
    orientations = [ASF_OP_0_ONLY, ASF_OP_90_ONLY, ASF_OP_180_ONLY, ASF_OP_270_ONLY]
    face_feature1 = None
    face_feature2 = None

    # 检测第一张图中的人脸
    for i, orientation in enumerate(orientations):
        print(f"Trying orientation {i + 1}/{len(orientations)} for the first image.")

        res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE, orientation, 30, 10, mask)
        if res != MOK:
            print("ASFInitEngine fail for orientation {}: {}".format(orientation, res))
            return None

        res, detectedFaces1 = face_engine.ASFDetectFaces(img1)
        if res == MOK and detectedFaces1.faceRect:
            single_detected_face1 = ASF_SingleFaceInfo()
            single_detected_face1.faceRect = detectedFaces1.faceRect[0]
            single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
            res, face_feature1 = face_engine.ASFFaceFeatureExtract(img1, single_detected_face1)
            if res == 90127:
                print("Detected specific error code 90127, skipping further attempts for the second image.")
                break
            if res == MOK:
                break  # 成功提取特征，退出循环
            else:
                print("ASFFaceFeatureExtract 1 fail: {}".format(res))
        else:
            print("ASFDetectFaces 1 fail for orientation {}: {}".format(orientation, res))

    if face_feature1 is None:
        print("No valid face feature extracted from the first image.")
        return None

    # 检测第二张图中的人脸
    for i, orientation in enumerate(orientations):
        print(f"Trying orientation {i + 1}/{len(orientations)} for the second image.")

        res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE, orientation, 30, 10, mask)
        if res != MOK:
            print("ASFInitEngine fail for orientation {}: {}".format(orientation, res))
            return None

        res, detectedFaces2 = face_engine.ASFDetectFaces(img2)

        # 检查特定错误代码
        if res == 90127:
            print("Detected specific error code 90127, skipping further attempts for the second image.")
            break

        if res == MOK and detectedFaces2.faceRect:
            single_detected_face2 = ASF_SingleFaceInfo()
            single_detected_face2.faceRect = detectedFaces2.faceRect[0]
            single_detected_face2.faceOrient = detectedFaces2.faceOrient[0]
            res, face_feature2 = face_engine.ASFFaceFeatureExtract(img2, single_detected_face2)
            if res == MOK:
                break  # 成功提取特征，退出循环
            else:
                print("ASFFaceFeatureExtract 2 fail: {}".format(res))
        else:
            print("ASFDetectFaces 2 fail for orientation {}: {}".format(orientation, res))

    if face_feature2 is None:
        print("No valid face feature extracted from the second image.")
        return None

    # 比较两个人脸的相似度
    res, score = face_engine.ASFFaceFeatureCompare(face_feature1, face_feature2)
    return score  # 返回相似度得分


# Define a POST endpoint
@app.post("/api/predict/facesmi")
async def post_facesim(
        image1: str = Body(embed=True,alias="image1", min_length=10),
        image2: str = Body(embed=True,alias="image2", min_length=10),
):
    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    try:
        # 检测第一张图中的人脸
        img1_ori = await read_image_from_url(image1)
        img2_ori = await read_image_from_url(image2)
        result = getfacesim(img1_ori, img2_ori)
        print(f"{image1} and {image2} sim is {result}")
        del img1_ori
        del img2_ori
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
