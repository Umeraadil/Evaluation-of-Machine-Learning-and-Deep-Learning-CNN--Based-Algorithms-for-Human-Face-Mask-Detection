# -*- coding: utf-8 -*-
"""
Created on Sun May 29 21:41:04 2022

@author: UMER AADIL
"""
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import skimage.feature as feature

# settings for LBP
radius = 1 
n_points = 8 * radius 


image = cv.imread('C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/DATASET/cropped datasets/crop color dataset/withmask/1_0_4.jpg')
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.imshow('image', image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
lbp = feature.local_binary_pattern(gray, n_points, radius)
print(lbp)

#cv.imwrite("C:/Users/dell/Desktop/new.jpg", lbp)
#cv.namedWindow("lbp", cv.WINDOW_NORMAL)
cv.imshow('lbp', lbp)
cv.waitKey(0)==ord('q')