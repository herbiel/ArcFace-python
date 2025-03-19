#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：face_service.py
@Author  ：herbiel
@Date    ：2024/11/21 18:15 
'''
from arcface.engine import *
import cv2
import logging

class FaceService:
    def __init__(self, app_id, sdk_key):
        self.app_id = app_id
        self.sdk_key = sdk_key
        self.face_engine = None
        self.mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER
        self.initialize()

    def initialize(self):
        # 激活接口
        res = ASFOnlineActivation(self.app_id, self.sdk_key)
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

        # 初始化引擎
        self.face_engine = ArcFace()
        res = self.face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 30, 10, self.mask)
        if (res != MOK):
            print("ASFInitEngine fail: {}".format(res))
        else:
            print("ASFInitEngine sucess: {}".format(res))

    def get_face_similarity(self, img1, img2, img1_url, img2_url):
        face_feature1 = None
        face_feature2 = None

        # 检测第一张图中的人脸
        res, detectedFaces1 = self.face_engine.ASFDetectFaces(img1)
        if res == MOK and detectedFaces1.faceRect:
            single_detected_face1 = ASF_SingleFaceInfo()
            single_detected_face1.faceRect = detectedFaces1.faceRect[0]
            single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
            res, face_feature1 = self.face_engine.ASFFaceFeatureExtract(img1, single_detected_face1)
            if res == 90127:
                print("Detected specific error code 90127, skipping further attempts for {}.".format(img1_url))
            elif res == 0:
                pass
            else:
                print("ASFFaceFeatureExtract 1 on {} fail : {}".format(img1_url, res))
        else:
            print("ASFDetectFaces 1 fail for on {}: {}".format(img1_url, res))

        if face_feature1 is None:
            print("No valid face feature extracted from the first image.")
            return None

        # 检测第二张图中的人脸
        res, detectedFaces2 = self.face_engine.ASFDetectFaces(img2)
        if res == MOK and detectedFaces2.faceRect:
            single_detected_face2 = ASF_SingleFaceInfo()
            single_detected_face2.faceRect = detectedFaces2.faceRect[0]
            single_detected_face2.faceOrient = detectedFaces2.faceOrient[0]
            res, face_feature2 = self.face_engine.ASFFaceFeatureExtract(img2, single_detected_face2)
            if res == 90127:
                print("Detected specific error code 90127, skipping further attempts for {}.".format(img2_url))
            elif res == 0:
                pass
            else:
                print("ASFFaceFeatureExtract 2 on {} fail : {}".format(img2_url, res))
        else:
            print("ASFDetectFaces 2 fail for on {}: {}".format(img2_url, res))

        if face_feature2 is None:
            print("No valid face feature extracted from the second image.")
            return None

        # 比较两个人脸的相似度
        res, score = self.face_engine.ASFFaceFeatureCompare(face_feature1, face_feature2)
        return score

    def get_face_info(self, img1, img1_url):
        res, detectedFaces1 = self.face_engine.ASFDetectFaces(img1)
        if res == MOK and detectedFaces1.faceRect:
            single_detected_face1 = ASF_SingleFaceInfo()
            single_detected_face1.faceRect = detectedFaces1.faceRect[0]
            single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
            res, face_feature1 = self.face_engine.ASFFaceFeatureExtract(img1, single_detected_face1)
            if res == 90127:
                print("Detected specific error code 90127, skipping further attempts for {}.".format(img1_url))
            elif res == 0:
                pass
            else:
                print("ASFFaceFeatureExtract 1 on {} fail : {}".format(img1_url, res))
        else:
            print("ASFDetectFaces 1 fail for on {}: {}".format(img1_url, res))

        if face_feature1 is None:
            print("No valid face feature extracted from the first image.")
            return None, None

        # 检测年龄和性别
        processMask = ASF_AGE | ASF_GENDER
        Gender, Age = None, None
        res = self.face_engine.ASFProcess(img1, detectedFaces1, processMask)
        if res == MOK:
            # 获取年龄
            res, ageInfo = self.face_engine.ASFGetAge()
            if (res != MOK):
                print("ASFGetAge fail: {}".format(res))
            else:
                print("Age: {}".format(ageInfo.ageArray[0]))
                Age = ageInfo.ageArray[0]

            # 获取性别
            res, genderInfo = self.face_engine.ASFGetGender()
            if (res != MOK):
                print("ASFGetGender fail: {}".format(res))
            else:
                print("Gender: {}".format(genderInfo.genderArray[0]))
                Gender = genderInfo.genderArray[0]
        else:
            Age = None
            Gender = None
        return Age, Gender

    def uninitialize(self):
        if self.face_engine:
            self.face_engine.ASFUninitEngine()
            self.face_engine = None

    def __del__(self):
        self.uninitialize()