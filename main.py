#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：main.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
from fastapi import FastAPI, Body, HTTPException
import requests
from io import BytesIO
from PIL import Image
import httpx
import cv2
import numpy as np
import threading
from arcface.engine import *
from config import APPID, SDKKey

app = FastAPI()

# Global variables for face engine and lock
face_engine = None
engine_lock = threading.Lock()


@app.on_event("startup")
async def startup_event():
    global face_engine

    # Activate the engine; requires internet access for the first time
    res = ASFOnlineActivation(APPID, SDKKey)
    if (MOK != res and MERR_ASF_ALREADY_ACTIVATED != res):
        print("ASFActivation fail: {}".format(res))
        return

    print("ASFActivation success: {}".format(res))

    # Get activation file info
    res, activeFileInfo = ASFGetActiveFileInfo()
    if (res != MOK):
        print("ASFGetActiveFileInfo fail: {}".format(res))
        return

    print(f"{activeFileInfo}")

    # Initialize face recognition engine
    face_engine = ArcFace()
    mask = ASF_FACE_DETECT | ASF_FACERECOGNITION

    # Initialize engine
    res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 30, 10, mask)
    if (res != MOK):
        print("ASFInitEngine fail: {}".format(res))
        return

    print("ASFInitEngine success: {}".format(res))


@app.on_event("shutdown")
async def shutdown_event():
    global face_engine
    if face_engine:
        face_engine.ASFUninitEngine()
        print("Face engine released.")


def getfacesim(img1, img2):
    orientations = [ASF_OP_0_ONLY, ASF_OP_90_ONLY, ASF_OP_180_ONLY, ASF_OP_270_ONLY]
    face_feature1 = None
    face_feature2 = None

    # Detect face in the first image
    for i, orientation in enumerate(orientations):
        print(f"Trying orientation {i + 1}/{len(orientations)} for the first image.")
        res, detectedFaces1 = face_engine.ASFDetectFaces(img1)
        if res == MOK and detectedFaces1.faceRect:
            single_detected_face1 = ASF_SingleFaceInfo()
            single_detected_face1.faceRect = detectedFaces1.faceRect[0]
            single_detected_face1.faceOrient = detectedFaces1.faceOrient[0]
            res, face_feature1 = face_engine.ASFFaceFeatureExtract(img1, single_detected_face1)
            if res == MOK:
                break  # Successfully extracted feature, exit loop
            else:
                print("ASFFaceFeatureExtract 1 fail: {}".format(res))
        else:
            print("ASFDetectFaces 1 fail for orientation {}: {}".format(orientation, res))

    if face_feature1 is None:
        print("No valid face feature extracted from the first image.")
        return None

    # Detect face in the second image
    for i, orientation in enumerate(orientations):
        print(f"Trying orientation {i + 1}/{len(orientations)} for the second image.")
        res, detectedFaces2 = face_engine.ASFDetectFaces(img2)

        if res == MOK and detectedFaces2.faceRect:
            single_detected_face2 = ASF_SingleFaceInfo()
            single_detected_face2.faceRect = detectedFaces2.faceRect[0]
            single_detected_face2.faceOrient = detectedFaces2.faceOrient[0]
            res, face_feature2 = face_engine.ASFFaceFeatureExtract(img2, single_detected_face2)
            if res == MOK:
                break  # Successfully extracted feature, exit loop
            else:
                print("ASFFaceFeatureExtract 2 fail: {}".format(res))
        else:
            print("ASFDetectFaces 2 fail for orientation {}: {}".format(orientation, res))

    if face_feature2 is None:
        print("No valid face feature extracted from the second image.")
        return None

    # Compare the similarity of the two faces
    res, score = face_engine.ASFFaceFeatureCompare(face_feature1, face_feature2)
    return score  # Return similarity score


async def read_image_from_url(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.post("/api/predict/facesmi")
async def post_facesim(
        image1: str = Body(embed=True, alias="image1", min_length=10),
        image2: str = Body(embed=True, alias="image2", min_length=10),
):
    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")

    try:
        img1_ori = await read_image_from_url(image1)
        img2_ori = await read_image_from_url(image2)

        with engine_lock:
            result = getfacesim(img1_ori, img2_ori)

        print(f"{image1} and {image2} sim is {result}")
        return {
            "code": 200,
            "error": None,
            "score": str(result)
        }
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
