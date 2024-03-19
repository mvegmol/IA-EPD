
import numpy as np
import pandas as pd

from cofiGradientFuncReg import cofiGradientFuncReg
from cofiCostFuncReg import cofiCostFuncReg

def checkNNGradients(lambda_param):
    #Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    #Zap out most entries
    Y = X_t @ Theta_t.T
    dim = Y.shape
    aux = np.random.rand(*dim)
    Y[aux > 0.5] = 0
    R = np.zeros((Y.shape))
    R[Y != 0] = 1

    #Run Gradient Checking
    dim_X_t = X_t.shape
    dim_Theta_t = Theta_t.shape
    X = np.random.randn(*dim_X_t)
    Theta = np.random.randn(*dim_Theta_t)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.concatenate((np.ravel(X,order='F'), np.ravel(Theta,order='F')))

    # Calculo gradiente mediante aproximación numérica
    mygrad = computeNumericalGradient(X, Theta, Y, R, num_features,lambda_param)

    #Calculo gradiente
    grad = cofiGradientFuncReg(params, Y, R, num_features,lambda_param)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    df = pd.DataFrame(mygrad,grad)
    print(df)

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm((mygrad-grad))/np.linalg.norm((mygrad+grad))

    print('If your gradient implementation is correct, then the differences will be small (less than 1e-9):' , diff)


def computeNumericalGradient(X,Theta, Y, R, num_features,lambda_param):
    mygrad = np.zeros(Theta.size + X.size)
    perturb = np.zeros(Theta.size + X.size)
    myeps = 0.0001
    params = np.concatenate((np.ravel(X, order='F'), np.ravel(Theta, order='F')))

    for i in range(np.size(Theta)+np.size(X)):
        # Set perturbation vector
        perturb[i] = myeps
        params_plus = params + perturb
        params_minus = params - perturb
        cost_high = cofiCostFuncReg(params_plus, Y, R, num_features,lambda_param)
        cost_low = cofiCostFuncReg(params_minus, Y, R, num_features,lambda_param)

        # Compute Numerical Gradient
        mygrad[i] = (cost_high - cost_low) / float(2 * myeps)
        perturb[i] = 0

    return mygrad
