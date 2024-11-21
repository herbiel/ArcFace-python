#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：check_face.py
@Author  ：herbiel
@Date    ：2024/11/15 10:53 
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：check_face.py
@Author  ：herbiel
@Date    ：2024/11/15 10:53 
'''
import cv2
import dlib
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def rotate_image(image, angle):
    """Rotate the image by the specified angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def detect_faces_dlib(image):
    """Detect faces using Dlib."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image)
    return faces


def load_image(image_source):
    """Load the image from a URL or local path."""
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


def find_faces_by_rotation(image_source):
    """Detect faces in the image, rotating if necessary."""
    image = load_image(image_source)
    if image is None:
        logging.error("Failed to load the image.")
        return None

    # Check for faces in the original image
    faces = detect_faces_dlib(image)
    if faces:
        logging.info(f"Detected {len(faces)} face(s) in the original image.")
        return annotate_faces(image, faces)

    # Rotate the image if no faces are detected
    for angle in range(90, 360, 90):
        rotated_image = rotate_image(image, angle)
        faces = detect_faces_dlib(rotated_image)
        if faces:
            logging.info(f"Detected {len(faces)} face(s) at {angle} degrees rotation.")
            return annotate_faces(rotated_image, faces)

    logging.warning("No faces detected even after rotating through all angles.")
    return None


def annotate_faces(image, faces):
    """Annotate detected faces with rectangles."""
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 调用函数，传入图像的 URL 或本地路径
# img = find_faces_by_rotation('https://monas-001.oss-ap-southeast-5.aliyuncs.com/image/025166536f5138cc493e71736ece4da1.jpeg')
# if img is not None:
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
# else:
#     print("No faces detected in the image.")
