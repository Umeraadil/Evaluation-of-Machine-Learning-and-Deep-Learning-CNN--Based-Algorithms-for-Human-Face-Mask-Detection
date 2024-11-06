# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 01:48:39 2022

@author:Umer Aadil
"""
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/dell/Desktop/FACEMASK/SVMMASK DECT/DATASET/full face datasets/gray color dataset',
        target_size=(224,224),
        batch_size=16 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/dell/Desktop/FACEMASK/CNN/merge datasets',
        target_size=(224,224),
        batch_size=16,
        class_mode='binary')

model_saved=model.fit_generator(
        training_set,
        epochs=1,
        validation_data=test_set,

        )

model.save('abcmodel.h5',model_saved)