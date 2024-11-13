#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：test.py
@Author  ：herbiel
@Date    ：2024/11/12 16:21 
'''
import requests
import json
import pymysql

# 数据库配置
db_config = {
    'host': 'gitlab.tangbull.com',  # 数据库主机
    'user': 'work',  # 数据库用户名
    'password': 'UdJBoYR+a/3u1pOLcq8lxA',  # 数据库密码
    'database': 'idn_engine'  # 数据库名称
}

# API 配置
api_url = 'http://149.129.236.15:8800/api/predict/facesmi'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

def fetch_images_from_db():
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # 查询 old_image 和 new_image
            sql = "SELECT old_image, new_image FROM local_face_test"
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        connection.close()

def update_score_in_db(old_image, new_image, new_score):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # 更新 new_score
            sql = "UPDATE local_face_test SET new_score = %s WHERE old_image = %s AND new_image = %s"
            cursor.execute(sql, (new_score, old_image, new_image))
        connection.commit()
    finally:
        connection.close()

def send_post_request(image1, image2):
    data = {
        "image1": image1,
        "image2": image2
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()  # 返回 JSON 格式的响应
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    # 从数据库中获取图像
    images = fetch_images_from_db()
    for old_image, new_image in images:
        image1_url = f"https://monas-001.oss-ap-southeast-5.aliyuncs.com/{old_image}"
        image2_url = f"https://monas-001.oss-ap-southeast-5.aliyuncs.com/{new_image}"

        # 发送 POST 请求
        result = send_post_request(image1_url, image2_url)
        if result and 'new_score' in result:
            new_score = result['new_score']
            # 更新数据库中的 new_score
            #update_score_in_db(old_image, new_image, new_score)
            print(f"Updated new_score for old_image: {old_image}, new_image: {new_image} to {new_score}")

if __name__ == "__main__":
    main()
