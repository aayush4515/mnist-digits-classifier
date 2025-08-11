import numpy as np
from mnist import (x_train, y_train)
from mnist import (x_test, y_test)

# nodes array, n[i] denotes the number of nodes in each layer
n = [784, 16, 16, 10]

# randomly initialize weights and biases for layers 1, 2 and 3

# dimension of weight matrix: no. of nodes in current layer * no. of nodes in prev layer
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])

# dimension of a bias matrix: no. of nodes in current layer * 1
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

# sigmoid function: squeezes each activations to a real number between 0 and 1
def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

# prepare data for training
def prepare_data():
    # load 1000 training data from a dataset of 60000 mnist numbers
    # each data in X is 28*28 pixels, a total of 784 numbers ranging from 0 to 255
    training_data = x_train[:1000]

    # for each of the digit in training data, convert it into single vector of 784 pixels and add into a matrix
    # each pixel is converted into a number between 0.0 and 1.0 by dividing by 255.0
    X = np.array([np.array(img).flatten() for img in training_data], dtype=np.float32) / 255.0       # shape: 1000 * 784

    # transpose X to convert to 784 * 1000 matrix, this is the 0th layer called A0
    A0 = X.T

    # load 1000 training data labels
    labels = y_train[:1000]

    '''each value in labels is the label for respective value in X
    for example, first data in X is label 5

    construct a matrix Y (labels) that has 10 nodes, 0-9, each signifying the respective digits
    according to the label, one of the node will be given the value 1 (node with label 5 in this case)
    other nodes will be given value 0
    main idea: 0 denotes: no probability of that label being the true digit
    1 denotes: 100% probability of that label being the true digit
    for example, 5 will be initialized probability of 1 for first image, because it denotes 5, rest of the nodes
    will be initialized with 0
    '''

    # array for feed forwarding, dimension: 10 * 784
    y = []
    for label in labels:
        arr = [0] * n[3]
        arr[label] = 1
        y.append(arr)

    # convert into a numpy array
    Y = np.array(y)

    return A0, Y

# feed forward calculation: returns y_hat -> prediction of the model
def feed_forward(A0):
    # layer 1 calculations
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    # layer 2 calculations
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    # layer 3 calculations (output layer)
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # y_hat: prediction of the model
    y_hat = A3

    return y_hat

# prepare data and feed forward
A0, Y = prepare_data()
y_hat = feed_forward(A0)

# TODO: implement backpropagation
def backpropagation():
    pass

