# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:14:41 2022

@author: UMER AADIL
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#%matplotlib inline
data = pd.read_csv("C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/LBP/LBP GRAY DATASET.CSV")
data.shape

print('dataset sampples')
print(data.head())

X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))