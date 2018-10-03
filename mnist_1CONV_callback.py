from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

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


class PlottingCallback(tflearn.callbacks.Callback):
    def __init__(self, model, x,
                 layers_to_observe=(),
                 kernels=5,
                 inputs=6):
        self.model = model
        self.x = x
        self.kernels = kernels
        self.inputs = inputs
        self.observers = [tflearn.DNN(l) for l in layers_to_observe]

    def on_epoch_end(self, training_state):
        outputs = [o.predict(self.x) for o in self.observers]

        for i in range(self.inputs):
            plt.figure(frameon=False)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            ix = 1
            for o in outputs:
                for kernel in range(self.kernels):
                    plt.subplot(len(outputs), self.kernels, ix)
                    plt.imshow(o[i, :, :, kernel])
                    plt.axis('off')
                    ix += 1
            plt.savefig('mnist_saved/mnist_1CONV_callback/outputs-for-image:%i-at-epoch:%i.png'
                        % (i, training_state.epoch))


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
model.fit(trainX, trainY, n_epoch=2, shuffle=True, validation_set=(testX, testY),
          show_metric=True,
          batch_size=20, run_id='mnist_1CONV_callback',
          callbacks=[PlottingCallback(model, testX, (network1, network2))])

# Save model when training is complete to a file
model.save('mnist_saved/mnist_1CONV_callback/mnistCV_1CONV_callback')

validx_pck = 'pickle/validX.pkl'
validy_pck = 'pickle/validY.pkl'

pickle.dump(validX, open(validx_pck, 'wb'))
pickle.dump(validY, open(validy_pck, 'wb'))
