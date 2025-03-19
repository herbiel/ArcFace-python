#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：detect_face_class.py
@Author  ：herbiel
@Date    ：2024/11/21 18:15 
'''
from arcface.engine import *
import cv2
import requests
import numpy as np
from config import APPID, SDKKey
from io import BytesIO
from PIL import Image
import logging

class FaceDetector:
    def __init__(self, app_id=APPID, sdk_key=SDKKey):
        self.app_id = app_id
        self.sdk_key = sdk_key
        self.face_engine = None
        self.mask = ASF_FACE_DETECT | ASF_FACERECOGNITION
        self.initialize()

    def initialize(self):
        # 激活接口,首次需联网激活
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

        # 获取人脸识别引擎
        self.face_engine = ArcFace()

        # 初始化接口
        res = self.face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 30, 10, self.mask)
        if (res != MOK):
            print("ASFInitEngine fail: {}".format(res))
        else:
            print("ASFInitEngine sucess: {}".format(res))

    def uninitialize(self):
        if self.face_engine:
            self.face_engine.ASFUninitEngine()
            self.face_engine = None

    def read_image_from_url(self, url):
        response = requests.get(url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    def load_image(self, image_source):
        try:
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source)
                response.raise_for_status()
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

    def get_face_feature_from_url(self, img_url):
        img = self.load_image(img_url)
        return self.get_face_feature(img, img_url)

    def get_face_feature(self, img, img_url):
        res, detectedFaces = self.face_engine.ASFDetectFaces(img)
        logging.error(f"img url {img_url} detectedFaces Info is : {detectedFaces}")
        
        if res == MOK:
            single_detected_face = ASF_SingleFaceInfo()
            single_detected_face.faceRect = detectedFaces.faceRect[0]
            single_detected_face.faceOrient = detectedFaces.faceOrient[0]
            res, face_feature = self.face_engine.ASFFaceFeatureExtract(img, single_detected_face)
            if (res != MOK):
                number = 0
                print("{} ASFFaceFeatureExtract fail: {}".format(img_url, res))
            else:
                number = 1
        else:
            number = 0
            print("{} ASFDetectFaces fail: {}".format(img_url, res))
        return number

    def extract_feature(self, image, img1_url):
        face_feature = None
        res, detectedFaces = self.face_engine.ASFDetectFaces(image)
        
        if res == MOK and detectedFaces.faceRect:
            single_detected_face = ASF_SingleFaceInfo()
            single_detected_face.faceRect = detectedFaces.faceRect[0]
            single_detected_face.faceOrient = detectedFaces.faceOrient[0]
            res, face_feature = self.face_engine.ASFFaceFeatureExtract(image, single_detected_face)
            if res == 90127:
                print("Detected specific error code 90127, skipping further attempts for {}.".format(img1_url,))
            elif res == 0:
                pass
            else:
                print("ASFFaceFeatureExtract 1 on {} fail : {}".format(img1_url, res))
        else:
            print("ASFDetectFaces 1 fail for on {}: {}".format(img1_url, res))

        if face_feature is None:
            print("No valid face feature extracted from the first image.")
        return face_feature

    def __del__(self):
        self.uninitialize()