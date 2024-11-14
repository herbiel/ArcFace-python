#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：arcface_detect.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
from  arcface.engine import *

from config import  APPID,SDKKey


import face_alignment
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,device='cpu',face_detector='blazeface')

def face_align_V2(input_image):

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
def getfacesim(img1,img2):
    #检测第一张图中的人脸
    res,detectedFaces1 = face_engine.ASFDetectFaces(img1)
    print(f"{detectedFaces1}")
    if res==MOK:
        single_detected_face1 = ASF_SingleFaceInfo()
        single_detected_face1.faceRect = detectedFaces1.faceRect[0]
        single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
        res ,face_feature1= face_engine.ASFFaceFeatureExtract(img1,single_detected_face1)
        if (res!=MOK):
            res, face_feature1 = face_engine.ASFFaceFeatureExtract(face_align_V2(img1), single_detected_face1)
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
            res, face_feature1 = face_engine.ASFFaceFeatureExtract(face_align_V2(img2), single_detected_face1)
            print ("ASFFaceFeatureExtract 2 fail: {}".format(res))
    else:
        print("ASFDetectFaces 2 fail: {}".format(res))

    #比较两个人脸的相似度
    res,score = face_engine.ASFFaceFeatureCompare(face_feature1,face_feature2)
    #print("相似度:",score)
    return score