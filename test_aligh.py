#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python
@File    ：test_aligh.py
@Author  ：herbiel
@Date    ：2024/11/12 16:46
'''
import cv2
import facealignment


# Read sample images
single_face = cv2.imread("asserts/2fc1cd543454583e0d4fa42578beb354.jpeg")

# Instantiate FaceAlignmentTools class
tool = facealignment.FaceAlignmentTools()

# MTCNN need RGB instead of CV2-BGR images
single_face = cv2.cvtColor(single_face, cv2.COLOR_BGR2RGB)

# Align image with single face
aligned_img = tool.align(single_face)
screen_img = cv2.hconcat([single_face, aligned_img])
screen_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)
cv2.imshow("Aligned Example Image", screen_img)
cv2.waitKey(0)