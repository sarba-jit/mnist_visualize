from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

import pickle

validx_pck = 'pickle/validX.pkl'
validy_pck = 'pickle/validY.pkl'

validX = pickle.load(open(validx_pck, 'rb'))
validY = pickle.load(open(validy_pck, 'rb'))

# Convolutional network building
# Define our network architecture:
network = input_data(shape=[None, 28, 28, 1])

# Step 1: Convolution
network1 = conv_2d(network, 10, filter_size=5, strides=1, activation='relu')
print(network1.shape, 'network1')

# Step 2: Max pooling
network2 = max_pool_2d(network1, kernel_size=2)
print(network2.shape, 'network2')

# Step 3: Fully-connected neural network with ten outputs
network3 = fully_connected(network2, 10, activation='softmax')
print(network3.shape, 'network3')

# Wrap the network in a model object
model = tflearn.DNN(network3, tensorboard_verbose=2)

model.load('mnist_saved/mnistCV_1CONV')

observed = [network1, network2, network3]
observers = [tflearn.DNN(v, session=model.session) for v in observed]
outputs = [m.predict(validX[0].reshape([-1, 28, 28, 1])) for m in observers]
print([d.shape for d in outputs])

print(model.predict(validX[0].reshape([-1, 28, 28, 1])))
