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
import cv2
import numpy as np
from check_face import find_faces_by_rotation
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




# Define a Pydantic model for the request body
def read_image_from_url(url):
    # Fetch the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    # Convert the image data to a NumPy array
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    # Decode the image array into an OpenCV format
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


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
            print("Detected specific error code 90127, skipping further attempts for {}.".format(img1_url,))
        elif res == 0:
            pass
        else:
            print("ASFFaceFeatureExtract 1 on {} fail : {}".format(img1_url,res))
    else:
        print("ASFDetectFaces 1 fail for on {}: {}".format(img1_url, res))

    if face_feature1 is None:
        print("No valid face feature extracted from the first image.")
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
            print("Detected specific error code 90127, skipping further attempts for {}.".format(img2_url, ))
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
    res, score = face_engine.ASFFaceFeatureCompare(face_feature1, face_feature2)
    return score  # 返回相似度得分

def check_face(img):
    res, detectedFaces = face_engine.ASFDetectFaces(img)
    if res==MOK:
        return detectedFaces.faceNum
    else:
        print("ASFDetectFaces  fail: {}".format(res))
        return None



# Define a POST endpoint
@app.post("/api/predict/facesmi")
async def post_facesim(
        image1: str = Body(embed=True,alias="image1", min_length=10),
        image2: str = Body(embed=True,alias="image2", min_length=10),
):
    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    try:
        img1_ori = read_image_from_url(image1)
        img2_ori = read_image_from_url(image2)

        facenumber1 = check_face(img1_ori)
        facenumber2 = check_face(img1_ori)
        rotation_image1 = None
        rotation_image2 = None
        if facenumber1 == 0:
            rotation_image1 = find_faces_by_rotation(image1)
        elif facenumber2 == 0:
            rotation_image2 = find_faces_by_rotation(image1)
        # if img1_ori is None or rotation_image1 is None:
        #     return {
        #         "code": 200,
        #         "error": "First image does not contain a detectable face",
        #         "score": None
        #     }
        # if img2_ori is None or rotation_image2 is None:
        #     return {
        #         "code": 200,
        #         "error": "Second image does not contain a detectable face",
        #         "score": None
        #     }
        if facenumber1 == 0:
            result = getfacesim(rotation_image1, img2_ori,image1,image2)
        elif facenumber2 == 0:
            result = getfacesim(img2_ori, rotation_image2, image1, image2)
        elif facenumber1 == 0 and facenumber2 == 0:
            result = getfacesim(rotation_image1, rotation_image2, image1, image2)
        else:
            result = getfacesim(img1_ori, img2_ori, image1, image2)
        print(f"{image1} and {image2} sim is {result}")
        output = {
                    "code": 200,
                    "error": None,
                    "score": str(result)
                }

        return output
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
