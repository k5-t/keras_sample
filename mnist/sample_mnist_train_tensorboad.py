#!/usr/bin/python
# -*- coding: utf-8 -*-

# imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
import numpy as np


batch_size = 128
num_classes = 10
epochs = 2

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# add for TensorBoard
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

# build model
model = Sequential()
model.add(Dense(512, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# define path
f_log = './log/'
f_model = './model'

tb_cb = TensorBoard(log_dir=f_log, histogram_freq=1)
cp_cb = ModelCheckpoint(filepath = os.path.join(f_model,'mnist_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
cbks = [tb_cb, cp_cb]
###

# fit model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger(f_log + 'training.log')
hist = model.fit(x_train, y_train, 
                   batch_size=batch_size, 
                   nb_epoch=epochs, 
                   verbose=1, 
                   callbacks=cbks, 
                   validation_data=(x_test, y_test))

# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])


# model saving
print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join(f_model,'mnist_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join(f_model,'mnit_model.yaml'), 'w').write(yaml_string)

# param saving
print('save weights')
model.save_weights(os.path.join(f_model,'mnist_model_weights.hdf5'))


# Prediction
from sklearn.metrics import confusion_matrix
import numpy as np

predict_classes = model.predict_classes(x_test[1:10,], batch_size=32)
true_classes = np.argmax(y_test[1:10],1)
print(confusion_matrix(true_classes, predict_classes))

### add for TensorBoard
KTF.set_session(old_session)
###
