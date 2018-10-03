from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle

import tflearn.datasets.mnist as mnist

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


# Convolutional network building
# Define our network architecture:
network = input_data(shape=[None, 28, 28, 1])

# Step 1: Convolution
network1 = conv_2d(network, 10, filter_size=5, strides=1, activation='relu')

# Step 2: Max pooling
network2 = max_pool_2d(network1, kernel_size=2)

# Step 3: Fully-connected neural network with ten outputs
network3 = fully_connected(network2, 10, activation='softmax')

# Tell tflearn how we want to train the network
network4 = regression(network3, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Train using classifier
# Wrap the network in a model object
model = tflearn.DNN(network4, tensorboard_verbose=2)
# Train it! We'll do 50 training passes and monitor it as it goes.
model.fit(trainX, trainY, n_epoch=1, shuffle=True, validation_set=(testX, testY),show_metric=True,
          batch_size=10, run_id='mnistCV_1CONV')


# Save model when training is complete to a file

model.save('mnist_saved/mnistCV_1CONV')

validx_pck = 'pickle/validX.pkl'
validy_pck = 'pickle/validY.pkl'

pickle.dump(validX, open(validx_pck, 'wb'))
pickle.dump(validY, open(validy_pck, 'wb'))