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
from io import BytesIO
from PIL import Image
import logging




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
def get_face_feature_from_url(img_url):
    img = load_image(img_url)
    res, detectedFaces = face_engine.ASFDetectFaces(img)

    logging.error(f"img url  {img_url} detectedFaces Info is : {detectedFaces}")
    if res == MOK:
        single_detected_face = ASF_SingleFaceInfo()
        single_detected_face.faceRect = detectedFaces.faceRect[0]
        single_detected_face.faceOrient = detectedFaces.faceOrient[0]
        res, face_feature = face_engine.ASFFaceFeatureExtract(img, single_detected_face)
        if (res != MOK):
            number = 0
            print("{} ASFFaceFeatureExtract  fail: {}".format(img_url,res))
        else:
            number = 1
    else:
        number = 0
        print("{} ASFFaceFeatureExtract  fail: {}".format(img_url,res))
    return number
def get_face_feature(img,img_url):
    # 检测第一张图中的人脸

    res, detectedFaces = face_engine.ASFDetectFaces(img)

    logging.error(f"img url  {img_url} detectedFaces Info is : {detectedFaces}")
    if res == MOK:
        single_detected_face = ASF_SingleFaceInfo()
        single_detected_face.faceRect = detectedFaces.faceRect[0]
        single_detected_face.faceOrient = detectedFaces.faceOrient[0]
        res, face_feature = face_engine.ASFFaceFeatureExtract(img, single_detected_face)
        if (res != MOK):
            number = 0
            print("{} ASFFaceFeatureExtract  fail: {}".format(img_url,res))
        else:
            number = 1
    else:
        number = 0
        print("{} ASFDetectFaces  fail: {}".format(img_url,res))
    return number


def extract_feature(image,img1_url):
    face_feature = None

    # 检测第一张图中的人脸


    res, detectedFaces = face_engine.ASFDetectFaces(image)
    if res == MOK and detectedFaces.faceRect:
        single_detected_face = ASF_SingleFaceInfo()
        single_detected_face.faceRect = detectedFaces.faceRect[0]
        single_detected_face.faceOrient = detectedFaces.faceOrient[0]
        res, face_feature = face_engine.ASFFaceFeatureExtract(image, single_detected_face)
        if res == 90127:
            print("Detected specific error code 90127, skipping further attempts for {}.".format(img1_url,))
        elif res == 0:
            pass
        else:
            print("ASFFaceFeatureExtract 1 on {} fail : {}".format(img1_url,res))
    else:
        print("ASFDetectFaces 1 fail for on {}: {}".format(img1_url, res))

    if face_feature is None:
        print("No valid face feature extracted from the first image.")
        return None