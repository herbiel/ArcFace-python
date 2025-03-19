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
from fastapi import FastAPI, Body, HTTPException, status
from check_face import find_faces_by_rotation
from face_service import FaceService
from config import APPID, SDKKey

app = FastAPI()
face_service = None

@app.on_event("startup")
async def startup_event():
    global face_service
    face_service = FaceService(APPID, SDKKey)

@app.on_event("shutdown")
async def shutdown_event():
    global face_service
    if face_service:
        face_service.uninitialize()

@app.post("/api/predict/facesmi")
async def post_facesim(
        image1: str = Body(embed=True, alias="image1", min_length=10),
        image2: str = Body(embed=True, alias="image2", min_length=10),
):
    if "oss-ap-southeast-5" in image1 or "oss-ap-southeast-5" in image2:
        image1 = image1.replace("oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5-internal.aliyuncs.com")
        image2 = image2.replace("oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5-internal.aliyuncs.com")

    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    
    try:
        img1_ori, img2_ori = None, None
        num1, num2 = 0, 0
        num1, img1_ori = find_faces_by_rotation(image1)
        num2, img2_ori = find_faces_by_rotation(image2)
        
        if num1 != 1 and num2 != 1:
            return {
                "code": 200,
                "error": "ALL image does not contain a detectable face",
                "score": None
            }
        elif num1 == 0:
            return {
                "code": 200,
                "error": "First image does not contain a detectable face",
                "score": None
            }
        elif num2 == 0:
            return {
                "code": 200,
                "error": "Second image does not contain a detectable face",
                "score": None
            }
        elif num1 == 1 and num2 == 1:
            result = face_service.get_face_similarity(img1_ori, img2_ori, image1, image2)
            print(f"{image1} and {image2} sim is {result}")
            return {
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
        gc.collect()

@app.post("/api/predict/GetFaceInfo")
async def post_faceinfo(
        image1: str = Body(embed=True, alias="image1", min_length=10),
):
    if "oss-ap-southeast-5" in image1:
        image1 = image1.replace("oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5-internal.aliyuncs.com")
    
    if not image1:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    
    try:
        num1, img1_ori = find_faces_by_rotation(image1)
        if num1 != 1:
            return {
                "code": 200,
                "error": "ALL image does not contain a detectable face",
                "score": None
            }
        
        Age, Gender = face_service.get_face_info(img1_ori, image1)
        return {
            "code": 200,
            "Age": Age,
            "Gender": Gender,
        }
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
    finally:
        del num1, img1_ori
        gc.collect()

@app.post("/check_status")
async def check(status_code=status.HTTP_200_OK):
    return 200