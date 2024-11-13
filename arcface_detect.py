#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：arcface_detect.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
from  arcface.engine import *

APPID = b''
SDKKey = b''

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
    print (f"{img1} and {img2} sim is {score}")
    return score