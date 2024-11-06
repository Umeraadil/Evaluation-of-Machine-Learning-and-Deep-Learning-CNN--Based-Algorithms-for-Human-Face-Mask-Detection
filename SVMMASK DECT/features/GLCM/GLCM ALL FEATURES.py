
"""
Created on Sun May 22 19:03:37 2022

@author: UMER AADIL
"""
from os import listdir
from os.path import isfile, join
import numpy
import cv2
#import matplotlib.pyplot as plt
import skimage.feature as feature
import csv
with open('C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/GLCM/WITHOUTMASKCOLOR.csv', 'w') as file:
    writer = csv.writer(file)       
    writer.writerow([ "dissimilarity", "homogeneity","contrast","energy","correlation","ASM","label"])
mypath =("C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/DATASET/cropped datasets/crop color dataset 4/withoutmask")
  
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype = object)
  
for n in range(0, len(onlyfiles)):
    i=0
    
    path = join(mypath,onlyfiles[n])
    images[n] = cv2.imread(join(mypath,onlyfiles[n]),
                           cv2.IMREAD_UNCHANGED)
      
    img = cv2.imread(path)
    image= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
      
    glcm = feature.greycomatrix(image, distances = [1], angles= [0], levels=256)
    dissimilarity = feature.greycoprops(glcm, 'dissimilarity')
    homogeneity = feature.greycoprops(glcm, 'homogeneity') 
    contrast = feature.greycoprops(glcm, 'contrast')
    energy = feature.greycoprops(glcm, 'energy')
    correlation = feature.greycoprops(glcm, 'correlation')
    ASM = feature.greycoprops(glcm, 'ASM')
    #hist = feature.greycoprops(glcm, 'hist')
      
    with open('C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/GLCM/WITHOUTMASKCOLOR.csv','a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([dissimilarity[0][0],homogeneity[0][0],contrast[0][0],energy[0][0],correlation[0][0],ASM[0][0],i])
        file.close() 