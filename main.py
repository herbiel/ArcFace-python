#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：main.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
import gc
import cv2
from fastapi import FastAPI,Body,HTTPException,status
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
    mask = ASF_FACE_DETECT | ASF_FACERECOGNITION|ASF_AGE | ASF_GENDER

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


# Define a POST endpoint
@app.post("/api/predict/facesmi")
async def post_facesim(
        image1: str = Body(embed=True,alias="image1", min_length=10),
        image2: str = Body(embed=True,alias="image2", min_length=10),
):
    if "oss-ap-southeast-5" in image1 or "oss-ap-southeast-5" in image2:
        image1 = image1.replace("oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5-internal.aliyuncs.com")
        image2 = image2.replace("oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5-internal.aliyuncs.com")
        #print(f"image1_url: {image1_url},image2_url: {image2_url}")
    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    try:
        img1_ori, img2_ori = None, None  # 先初始化变量，防止 UnboundLocalError
        num1 = 0
        num2 = 0
        num1,img1_ori = find_faces_by_rotation(image1)
        num2,img2_ori = find_faces_by_rotation(image2)
        if num1 != 1 and num2 != 1:
            return {
                "code": 200,
                "error": "ALL image does not contain a detectable face",
                "score": None
            }
        elif num1 == 0:
            return  {
                "code": 200,
                "error": "First image does not contain a detectable face",
                "score": None
            }
        elif num2 == 0:
            return  {
                "code": 200,
                "error": "Second image does not contain a detectable face",
                "score": None
            }
        elif num1 == 1 and num2 == 1:
            result = getfacesim(img1_ori, img2_ori,image1,image2)
            print(f"{image1} and {image2} sim is {result}")
            return  {
                        "code": 200,
                        "error": "None",
                        "score": str(result)
                    }
        else:
            return {
                "code": 500,
                "error": "Face Detect Fail",
                "score": None
            }

    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
    finally:

        del num1, num2, img1_ori, img2_ori
        gc.collect()  # 手动释放内存
@app.post("/check_status")
async def check(status_code=status.HTTP_200_OK):
    return 200


def getfaceinfo(img1,img1_url):
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
    # 检测第一张图中的人脸
    processMask = ASF_AGE | ASF_GENDER
    Gender,Age = None,None
    res = face_engine.ASFProcess(img1,detectedFaces1,processMask)
    if res == MOK:
        # 获取年龄
        res, ageInfo = face_engine.ASFGetAge()
        if (res != MOK):
            print("ASFGetAge fail: {}".format(res))
        else:
            print("Age: {}".format(ageInfo.ageArray[0]))
            Age = ageInfo.ageArray[0]

        # 获取性别
        res, genderInfo = face_engine.ASFGetGender()
        if (res != MOK):
            print("ASFGetGender fail: {}".format(res))
        else:
            print("Gender: {}".format(genderInfo.genderArray[0]))
            Gender = genderInfo.genderArray[0]
    else:
        Age = None,
        Gender = None
    return Age,Gender


@app.post("/api/predict/GetFaceInfo")
async def post_faceinfo(
        image1: str = Body(embed=True,alias="image1", min_length=10),
):
    if "oss-ap-southeast-5" in image1 :
        image1 = image1.replace("oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5-internal.aliyuncs.com")
        #print(f"image1_url: {image1_url},image2_url: {image2_url}")
    if not image1:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    try:
        num1,img1_ori = find_faces_by_rotation(image1)
        if num1 != 1:
            return {
                "code": 200,
                "error": "ALL image does not contain a detectable face",
                "score": None
            }
        Age,Gender = getfaceinfo(img1_ori,image1)
        return {
            "code": 200,
            "Age": Age,
            "Gende": Gender,
        }
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
    finally:

        del num1,  img1_ori
        gc.collect()  # 手动释放内存
