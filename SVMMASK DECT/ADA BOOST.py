# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 21:26:51 2022

@author: ZAHID
"""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#%matplotlib inline
data = pd.read_csv("C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/LBP/LBP COLOR DATASET 1.CSV")
data.shape

print('dataset sampples')
print(data.head())

X = data.drop('label', axis=1)
#print(X)
y = data['label']
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
Ab =  AdaBoostClassifier(n_estimators=100, random_state=6)
Ab.fit(X_train, y_train)
y_pred = Ab.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))