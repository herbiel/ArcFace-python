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
from check_face import find_faces_by_rotation

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




# Define a Pydantic model for the request body

def getfacesim(img1, img2,img1_url,img2_url):
    face_feature1 = None
    face_feature2 = None

    # 检测第一张图中的人脸


    res, detectedFaces1 = face_engine.ASFDetectFaces(img1)
    if res == MOK and detectedFaces1.faceRect:
        single_detected_face1 = ASF_SingleFaceInfo()
        single_detected_face1.faceRect = detectedFaces1.faceRect[0]
        single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
        res, face_feature1 = face_engine.ASFFaceFeatureExtract(img1, single_detected_face1)
        if res == 90127:
            print("Detected specific error code 90127, skipping further attempts for {}.".format(img1_url))
        else:
            print("ASFFaceFeatureExtract 1 fail on {} res is {}".format(img1_url,res))
    else:
        print("ASFDetectFaces 1 fail for  {}: {}".format(img1_url,res))

    if face_feature1 is None:
        print("No valid face feature extracted from {}.".format(img1_url))
        return None

    # 检测第二张图中的人脸


    res, detectedFaces2 = face_engine.ASFDetectFaces(img2)

    # 检查特定错误代码
    if res == 90127:
        print("Detected specific error code 90127, skipping further attempts for the second image.")

    if res == MOK and detectedFaces2.faceRect:
        single_detected_face2 = ASF_SingleFaceInfo()
        single_detected_face2.faceRect = detectedFaces2.faceRect[0]
        single_detected_face2.faceOrient = detectedFaces2.faceOrient[0]
        res, face_feature2 = face_engine.ASFFaceFeatureExtract(img2, single_detected_face2)
        if res == 90127:
            print("Detected specific error code 90127, skipping further attempts for {}.".format(img2_url))
        else:
            print("ASFFaceFeatureExtract 2 fail on {} res is {}".format(img2_url, res))
    else:
        print("ASFDetectFaces 2 fail for  {}: {}".format(img2_url, res))

    if face_feature1 is None:
        print("No valid face feature extracted from {}.".format(img2_url))
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
        output = {
                "code": 200,
                "error": None,
                "score": 0
            }
        # 检测第一张图中的人脸
        img1_ori = find_faces_by_rotation(image1)
        img2_ori = find_faces_by_rotation(image2)
        if not img1_ori or not img2_ori:
            result = getfacesim(img1_ori ,img2_ori,image1,image2)
            print(f"{image1} and {image2} sim is {result}")
            output = {
                "code": 200,
                "error": "One of them without face",
                "score": str(result)
            }
        else:
            output = {
                "code": 200,
                "error": "One of them without face",
                "score": 0
            }
        del img1_ori
        del img2_ori
        return output
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
