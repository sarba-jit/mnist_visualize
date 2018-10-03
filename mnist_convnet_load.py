# -*- coding: utf-8 -*-

## VK: I need to run this after I run resize_image.py

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist
import numpy as np
import pickle

validx_pck = 'pickle/validX.pkl'
validy_pck = 'pickle/validY.pkl'

validX = pickle.load(open(validx_pck, 'rb'))
validY = pickle.load(open(validy_pck, 'rb'))

input_layer = input_data(shape=[None, 28, 28, 1])
conv_layer  = conv_2d(input_layer, nb_filter=20, filter_size=5,
                      activation='sigmoid', name='conv_layer_1')
pool_layer  = max_pool_2d(conv_layer, 2, name='pool_layer_1')
fc_layer_1  = fully_connected(pool_layer, 100, activation='sigmoid',
                              name='fc_layer_1')
fc_layer_2 = fully_connected(fc_layer_1, 10, activation='softmax',
                             name='fc_layer_2')

model = tflearn.DNN(fc_layer_2)
model.load('mnist_saved/mnist_25CONV/my_net25')
for i in range(50):
    # print(model.predict(validX[0].reshape([-1, 28, 28, 1])))
    print(np.argmax(model.predict(validX[i].reshape([-1, 28, 28, 1])),axis=1), np.argmax(validY[i]))