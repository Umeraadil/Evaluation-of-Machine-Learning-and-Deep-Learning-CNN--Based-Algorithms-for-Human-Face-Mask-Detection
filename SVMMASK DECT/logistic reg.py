# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:34:19 2022

@author: UMER AADIL
"""
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/features/LBP/LBP GRAY DATASET.CSV")
data.shape

print('dataset sampples')
print(data.head())

X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
clf =LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#define the predictor variable and the response variable
x = data["label"]
y = data[""]

#plot logistic regression curve
sns.regplot(x=x, y=y, data=data, logistic=True, ci=None)