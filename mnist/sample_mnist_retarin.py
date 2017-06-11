#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import RMSprop
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
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

old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)

    json_string = open(os.path.join(f_model, model_filename)).read()
    print('load model:' + json_string)
    model = model_from_json(json_string)

    model.summary()

    # fit model
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    json_weight_string = os.path.join(f_model,weights_filename)
    print('load weights:' + json_weight_string)
    model.load_weights(json_weight_string)

    cbks = []

    csv_logger = CSVLogger(f_log + 'training.log')   

    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=cbks, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    f_model = './model'
    # model saving
    print('save the architecture of a model')
    json_string = model.to_json()
    open(os.path.join(f_model,'re_mnist_model.json'), 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(f_model,'re_mnit_model.yaml'), 'w').write(yaml_string)

    # param saving
    print('save weights')
    model.save_weights(os.path.join(f_model,'re_mnist_model_weights.hdf5'))

KTF.set_session(old_session)
