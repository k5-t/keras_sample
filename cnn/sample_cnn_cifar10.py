#!/usr/bin/python
# -*- coding: utf-8 -*-

# imports
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D


batch_size = 250
num_classes = 10
epochs = 2

# dimention
img_rows, img_cols = 32, 32

# channels
img_channels = 3

# load mnist
# (num_samples, nb_rows, nb_cols, nb_channel) = tf
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# change pix_value 0-255 => 0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# change class-labels 0-9 => type of one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# add for TensorBoard
#from keras.callbacks import TensorBoard, ModelCheckpoint
#import keras.backend.tensorflow_backend as KTF
#import tensorflow as tf

#old_session = KTF.get_session()

#session = tf.Session('')
#KTF.set_session(session)
#KTF.set_learning_phase(1)
###

# build model (CNN)
# CNNを構築
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


# fit model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# define path
f_log = './log/'
f_model = './model'

# prepare generating TesnorBoard
#tb_cb = TensorBoard(log_dir=f_log, histogram_freq=1)
#cp_cb = ModelCheckpoint(filepath = os.path.join(f_model,'mnist_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#cbks = [tb_cb, cp_cb]
###


es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger(f_log + 'training.log')

hist = model.fit(x_train, y_train, 
                   batch_size=batch_size, 
                   nb_epoch=epochs, 
                   verbose=1, 
#                   callbacks=cbks, 
                   validation_data=(x_test, y_test),
                   validation_split=0.1)

# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])


# model saving
print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join(f_model,'cnn_for_cifar10_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join(f_model,'cnn_for_cifar10_model.yaml'), 'w').write(yaml_string)

# param saving
print('save weights')
model.save_weights(os.path.join(f_model,'cnn_for_cifar10__model_weights.hdf5'))



# Prediction
#from sklearn.metrics import confusion_matrix
#import numpy as np

#predict_classes = model.predict_classes(x_test[1:10,], batch_size=32)
#true_classes = np.argmax(y_test[1:10],1)
#print(confusion_matrix(true_classes, predict_classes))

### add for TensorBoard
#KTF.set_session(old_session)
###
