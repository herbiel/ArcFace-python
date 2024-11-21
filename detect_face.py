#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：detect_face.py
@Author  ：herbiel
@Date    ：2024/11/21 18:15 
'''
from  arcface.engine import *
import cv2
from arcface.engine import *
import requests
import numpy as np
from config import  APPID,SDKKey


# 激活接口,首次需联网激活
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


def read_image_from_url(url):
    # Fetch the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Convert the image data to a NumPy array
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    # Decode the image array into an OpenCV format
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


def detect_face_number_from_url(img_url):
    img = read_image_from_url(img_url)
    # 检测第一张图中的人脸
    res, detectedFaces1 = face_engine.ASFDetectFaces(img)


    print(f"{detectedFaces1}")
    if res == MOK:
        faceNum = detectedFaces1.faceNum
        if (res != MOK):
            print("ASFFaceFeatureExtract {}  fail: {}".format(img_url,res))
    else:
        faceNum = 0
        print("ASFDetectFaces {} fail: {}".format(img_url,res))
    return faceNum

def detect_face_number(img,img_url):
    # 检测第一张图中的人脸
    res, detectedFaces = face_engine.ASFDetectFaces(img)

    if res != MOK:
        detectedFaces = 0
        print("ASFDetectFaces {} fail: {}".format(img_url,res))
    return detectedFaces