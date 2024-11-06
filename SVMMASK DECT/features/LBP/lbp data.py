# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 01:12:25 2022

@author: UMER AADIL
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2 
import skimage.feature as feature
import csv
with open('C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/LBP/fullWITHoutMASKGRAY.CSV', 'w') as file:
    writer = csv.writer(file)       
    writer.writerow(["label"])
mypath =("C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/DATASET/full face datasets/color dataset 2/without_mask")
  
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype = object)
  
for n in range(0, len(onlyfiles)):
    i=0
    numPoints = 24
    radius = 2
    path = join(mypath,onlyfiles[n])
    images[n] = cv2.imread(join(mypath,onlyfiles[n]),
                           cv2.IMREAD_UNCHANGED)
      
    img = cv2.imread(path)
    image= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
      
    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3),
       range=(0, numPoints + 2))
    fd=lbp.flatten()
    #cv2.imshow('lbp', lbp)
    #cv2.waitKey(0)==ord('q')
    #print(hist)
    #print(fd)
     
   # optionally normalize the histogram
    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    #a = np.asarray(([ hist ])+1)
    #np.savetxt("C:/Users/dell/Desktop/umm123.csv", a, delimiter=",")
    print(np.size(hist))
    print(hist)
    #pd.read_csv(r'C:/Users/dell/Desktop/wert123.csv', header=None, skiprows=[0])
    #plt.hist(fd)
    #plt.show()
    with open('C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/LBP/fullWITHoutMASKGRAY.CSV','a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([i,hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6],hist[7], hist[8], hist[9], hist[10], hist[11], hist[12], hist[13], hist[14], hist[15], hist[16], hist[17], hist[18], hist[19], hist[20],hist[21], hist[22], hist[23], hist[24], hist[25]])
        
        
        file.close() 
