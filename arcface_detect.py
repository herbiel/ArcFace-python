#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：arcface_detect.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
import cv2
from  arcface.engine import *
import requests
from aligh import aligh_image
import numpy as np
APPID = b'4oWW5VkWg5BiqejxL2y2bzTSkgvZZefKrvR9RaHTgHJ1'
SDKKey = b'6mwAL6TdSfRi1BmcEPjuGha8jFSS3mcygu25nLhz1kFq'

#激活接口,首次需联网激活
res = ASFOnlineActivation(APPID, SDKKey)
if (MOK != res and MERR_ASF_ALREADY_ACTIVATED != res):
    print("ASFActivation fail: {}".format(res))
else:
    print("ASFActivation sucess: {}".format(res))

# 获取激活文件信息
res,activeFileInfo = ASFGetActiveFileInfo()

if (res != MOK):
    print("ASFGetActiveFileInfo fail: {}".format(res))
else:
    print(f"{activeFileInfo}")

# 获取人脸识别引擎
face_engine = ArcFace()

# 需要引擎开启的功能
mask = ASF_FACE_DETECT | ASF_FACERECOGNITION

# 初始化接口
res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE,ASF_OP_0_ONLY,30,10,mask)
if (res != MOK):
    print("ASFInitEngine fail: {}".format(res) )
else:
    print("ASFInitEngine sucess: {}".format(res))

def read_image_from_url(url):
    try:
        # Fetch the image from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Convert the image data to a NumPy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # Decode the image array into an OpenCV format
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Decoding the image failed, resulting in None.")

        return image

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the image from the URL: {e}")
    except ValueError as e:
        print(f"Error processing the image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None

def getfacesim(img1_url,img2_url):
    img1_ori = read_image_from_url(img1_url)
    img2_ori = read_image_from_url(img2_url)
    img1 = aligh_image(img1_ori)
    img2 = aligh_image(img2_ori)
    #检测第一张图中的人脸
    res,detectedFaces1 = face_engine.ASFDetectFaces(img1)
    print(f"{detectedFaces1}")
    if res==MOK:
        single_detected_face1 = ASF_SingleFaceInfo()
        single_detected_face1.faceRect = detectedFaces1.faceRect[0]
        single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
        res ,face_feature1= face_engine.ASFFaceFeatureExtract(img1,single_detected_face1)
        if (res!=MOK):
            print ("ASFFaceFeatureExtract 1 fail: {}".format(res))
    else:
        print("ASFDetectFaces 1 fail: {}".format(res))

    #检测第二张图中的人脸
    res,detectedFaces2 = face_engine.ASFDetectFaces(img2)
    if res==MOK:
        single_detected_face2 = ASF_SingleFaceInfo()
        single_detected_face2.faceRect = detectedFaces2.faceRect[0]
        single_detected_face2.faceOrient = detectedFaces2.faceOrient[0]
        res ,face_feature2= face_engine.ASFFaceFeatureExtract(img2,single_detected_face2)
        if (res==MOK):
            pass
        else:
            print ("ASFFaceFeatureExtract 2 fail: {}".format(res))
    else:
        print("ASFDetectFaces 2 fail: {}".format(res))

    #比较两个人脸的相似度
    res,score = face_engine.ASFFaceFeatureCompare(face_feature1,face_feature2)
    #print("相似度:",score)
    print (f"{img1_url} and {img2_url} sim is {score}")
    return score