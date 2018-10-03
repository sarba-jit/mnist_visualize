# -*- coding: utf-8 -*-

## VK: I need to run this after I run resize_image.py

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist
import pickle

# X is the training data set; Y is the labels for X.
# testX is the testing data set; testY is the labels for testX.
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y = shuffle(X, Y)
trainX = X[0:50000]
trainY = Y[0:50000]
validX = X[50000:]
validY = Y[50000:]

trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

input_layer = input_data(shape=[None, 28, 28, 1])
print(input_layer.shape, 'input_layer')

conv_layer  = conv_2d(input_layer, nb_filter=20, filter_size=5,
                      activation='sigmoid', name='conv_layer_1')
print(conv_layer.shape, 'conv_layer')

pool_layer  = max_pool_2d(conv_layer, 2, name='pool_layer_1')
print(pool_layer.shape, 'pool_layer')

fc_layer_1  = fully_connected(pool_layer, 100, activation='sigmoid',
                              name='fc_layer_1')
print(fc_layer_1.shape, 'fc_layer_1')

fc_layer_2 = fully_connected(fc_layer_1, 10, activation='softmax',
                             name='fc_layer_2')
print(fc_layer_2.shape, 'fc_layer_2')

network = regression(fc_layer_2, optimizer='sgd',
                     loss='categorical_crossentropy',
                     learning_rate=0.1)

model = tflearn.DNN(network)
model.fit(trainX, trainY, n_epoch=25, shuffle=True, validation_set=(testX, testY),
          show_metric=True, batch_size=10, run_id='MNIST_ConvNet_25')
model.save('mnist_saved/mnist_25CONV/my_net25')

validx_pck = 'pickle/validX.pkl'
validy_pck = 'pickle/validY.pkl'

pickle.dump(validX, open(validx_pck, 'wb'))
pickle.dump(validY, open(validy_pck, 'wb'))