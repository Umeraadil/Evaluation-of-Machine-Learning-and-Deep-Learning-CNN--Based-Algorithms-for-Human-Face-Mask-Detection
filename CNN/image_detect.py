# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 02:28:29 2022

@author: ZAHID
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
#import argparse
import cv2
import joblib
#import os
#import copy



print("Please wait loading face detector model...")
prototxtPath =  r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print( "Please wait loading face mask detector model...")
mymodel =load_model('C:/Users/dell/Desktop/FACEMASK/CNN/models/avg_pooling_model.h5')

image = cv2.imread("C:/Users/dell/Desktop/FACEMASK/CNN/test/test5.jpg")
orig = image.copy()
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
print(" Please Wait while computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.2:

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
    
        (with_mask, without_mask) = mymodel.predict(face)[0]
        print("faces avalable in the image at", confidence)
        label = "Mask" if with_mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(with_mask, without_mask) * 100)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
cv2.imshow("Output", image)
cv2.waitKey(0)