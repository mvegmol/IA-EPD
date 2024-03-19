# CHECKNNGRADIENTS Creates a small neural network to check the
# backpropagation gradients. It will output the analytical gradients
# produced by your backprop code and the numerical gradients (computed
# using computeNumericalGradient). These two gradient computations should
# result in very similar values.
#

import numpy as np
import math as mt
import pandas as pd

from nnGradFunctionSinReg import nnGradFunctionSinReg
from nnCostFunctionSinReg import nnCostFunctionSinReg

def checkNNGradients(lambda_param):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m,input_layer_size-1)
    y = np.zeros(m)
    for i in range(m):
        y[i] = (1 + mt.fmod(i+1,num_labels))

    # Unroll parameters
    nn_params = np.hstack((Theta1.ravel(order='F'), Theta2.ravel(order='F')))

    # Compute back-propagation gradients
    nn_backprop_params = nnGradFunctionSinReg(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)

    # Compute gradient by numerical approximation
    mygrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels,X, y)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    df = pd.DataFrame(mygrad,nn_backprop_params)
    print(df)

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.py, then diff below should be less than 1e-9
    diff = np.linalg.norm((mygrad-nn_backprop_params))/np.linalg.norm((mygrad+nn_backprop_params))

    print('If your backpropagation implementation is correct, then the differences will be small (less than 1e-9):' , diff)


def debugInitializeWeights(fan_out, fan_in):

    # Set W to zeros
    W = np.zeros((fan_out,1+fan_in))

    # Initialize W using "sin", this ensures that W is always of the same values and will be useful for debugging
    b = np.zeros(W.size)
    for i in np.array(range(1,W.size+1)):
        b[i-1] = mt.sin(i)

    W = np.reshape(b,W.shape,order='F') / 10

    return W


def computeNumericalGradient(theta, input_layer_size, hidden_layer_size, num_labels,X, y):
    mygrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    myeps = 0.0001

    for i in range(np.size(theta)):
        # Set perturbation vector
        perturb[i] = myeps
        cost_high = nnCostFunctionSinReg(theta + perturb, input_layer_size,
                                         hidden_layer_size, num_labels,X, y)
        cost_low = nnCostFunctionSinReg(theta - perturb, input_layer_size,
                                        hidden_layer_size, num_labels,X, y)

        # Compute Numerical Gradient
        mygrad[i] = (cost_high - cost_low) / float(2 * myeps)
        perturb[i] = 0

    return mygrad