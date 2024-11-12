#!/usr/bin/env python
# -*- coding: UTF-8 -*-33.jpeg
# 44.jpeg
'''
@Project ：ArcFace-python 
@File    ：test_deepface.py
@Author  ：herbiel
@Date    ：2024/11/12 18:01 
'''
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]
backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'fastmtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

# result = DeepFace.verify(
#   img1_path ="asserts/e60d70f679c81a112b1d2ca0bcfa992b.jpeg",
#   img2_path ="asserts/75366cc59852f9ffb5366a53d3f7bb41.jpeg",
#   #enforce_detection=False,
#   model_name="Facenet",
#   detector_backend="retinaface"
#
# )
# print(result)

img1_path ="asserts/e60d70f679c81a112b1d2ca0bcfa992b.jpeg"

img1_data = cv2.imread(img1_path)
img1 = DeepFace.detectFace(img_path=img1_path,align=True,detector_backend="retinaface",enforce_detection=False)
plt.imshow(img1)

plt.show()