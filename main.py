#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ArcFace-python 
@File    ：main.py
@Author  ：herbiel
@Date    ：2024/11/12 15:31 
'''
from fastapi import FastAPI,Body,HTTPException
from pydantic import BaseModel
from arcface_detect import getfacesim
# Create a FastAPI instance
app = FastAPI()

# Define a Pydantic model for the request body
class Item(BaseModel):
    image1: str
    image2: str


# Define a POST endpoint
@app.post("/api/predict/facesmi")
def post_facesim(
        image1: str = Body(embed=True,alias="image1", min_length=10),
        image2: str = Body(embed=True,alias="image2", min_length=10),
):
    if not image1 or not image2:
        raise HTTPException(status_code=422, detail="Request Error, invalid image")
    try:
        result = getfacesim(image1, image2)
        return result
    except Exception as e:
        return {
            "code": 500,
            "error": str(e),
            "score": None
        }
