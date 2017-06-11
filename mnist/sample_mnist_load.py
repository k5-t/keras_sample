#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import RMSprop
import keras.callbacks
import os.path
from keras.callbacks import EarlyStopping, CSVLogger


batch_size = 128
nb_classes = 10
nb_epoch = 5

f_log = './log'
f_model = './model'
model_filename = 'mnist_model.json'
weights_filename = 'mnist_model_weights.hdf5'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

json_string = open(os.path.join(f_model, model_filename)).read()
print('load model:' + json_string)
model = model_from_json(json_string)

model.summary()

json_weight_string = os.path.join(f_model,weights_filename)
print('load weights:' + json_weight_string)
model.load_weights(json_weight_string)

# Prediction
import numpy as np
from sklearn.metrics import confusion_matrix


# Prediction
predict_classes = model.predict_classes(x_test[1:10,], batch_size=32)
true_classes = np.argmax(y_test[1:10],1)
print(confusion_matrix(true_classes, predict_classes))

