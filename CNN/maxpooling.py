# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 02:44:06 2022

@author: 
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

INIT_LR = 1e-4
EPOCHS = 10
BS = 16

DIRECTORY = r"C:/Users/dell/Desktop/FaceMaskDetection-main/dataset1"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)
for img in os.listdir(path):
 img = cv2.imread(os.path.join(path,img))
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
 #plt.show()
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


trainData, testData, trainLabel, testLabel = train_test_split(data, labels, test_size = 0.1)

model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (3, 3), input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

print("[INFO] compiling model...")
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('model/m-{epoch:03d}.model', monitor = 'val_loss', verbose = 0, save_best_only = True)

print("[INFO] training model...")
history = model.fit(trainData, trainLabel, epochs = 1, validation_split = 0.2)

print("[INFO] saving model...")
model.save("maxpooling_model.h5")


