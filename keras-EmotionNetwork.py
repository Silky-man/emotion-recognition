# -*- coding:utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import layers
from keras.optimizers import SGD
from datetime import datetime
# import cv2
# import numpy as np
# import os
# from keras import regularizers
# from matplotlib import pyplot as plt
TIMESTAMP = "{0:%Y-%m-%dTime%H-%M-%S/}".format(datetime.now())
train_batch_size=128
test_batch_size=1191
#datagen = ImageDataGenerator(rotation_range=15,horizontal_flip=True)
datagen2 = ImageDataGenerator()
train_data=datagen2.flow_from_directory(r'E:\dataset\lunwenyong\kerasyong\raf',
                                  batch_size=train_batch_size,
                                  target_size=(56,56),
                                  shuffle=True)
test_data=datagen2.flow_from_directory(r'E:\dataset\lunwenyong\kerasyong\ck',
                                  batch_size=test_batch_size,
                                  target_size=(56,56),
                                  shuffle=False)
print(test_data.class_indices)
#di:0   ha:1
#an:0  di:1  ha:2  ne:3

classes=7

def activation(x, func='relu'):
    return layers.Activation(func)(x)

def squeeze_excitation_layer(x, out_dim,ratio=32):
    squeeze = layers.GlobalAveragePooling2D()(x)
    excitation = layers.Dense(units=out_dim // ratio)(squeeze)
    excitation = activation(excitation)
    excitation = layers.Dense(units=out_dim)(excitation)
    excitation = activation(excitation, 'sigmoid')
    excitation = layers.Reshape((1, 1, out_dim))(excitation)
    scale = layers.multiply([x, excitation])
    return scale

input=layers.Input(shape=(56,56,3))

model=layers.Conv2D(64,(3,3),strides=(1,1),input_shape=(56,56,3),padding='same')(input)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model = squeeze_excitation_layer(model, 64)

model = layers.Conv2D(64, (3, 3), strides=(1,1), padding='same')(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model = squeeze_excitation_layer(model, 64)
model = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(model)

model = layers.Conv2D(256, (3, 3), strides=(1,1), padding='same')(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model = squeeze_excitation_layer(model, 256)
model = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(model)

model = layers.Conv2D(256, (3, 3), strides=(1,1), padding='same')(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model = squeeze_excitation_layer(model, 256)

model = layers.Conv2D(128, (3, 3), strides=(1,1), padding='same')(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model = squeeze_excitation_layer(model, 128)

model = layers.Conv2D(128, (3, 3), strides=(1,1), padding='same')(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model = squeeze_excitation_layer(model, 128)
model=layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(model)

model=layers.Flatten()(model)
model=layers.Dense(2048)(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
model=layers.Dense(256)(model)
model = layers.BatchNormalization()(model)
model=layers.Activation('relu')(model)
output=layers.Dense(classes,activation='softmax')(model)
model=Model(inputs=input,outputs=output)
#print(model.summary())

sgd=SGD(lr=0.008,decay=0.0001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='E:\\tensorboard\\ck\\'+TIMESTAMP)
model.fit_generator(train_data,
                    steps_per_epoch=(train_data.samples//train_batch_size),
                    epochs=15,
                    validation_data=test_data,
                    validation_steps=(test_data.samples//test_batch_size),
                    callbacks=[tensorboard])
