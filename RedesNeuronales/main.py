import numpy as np
import scipy.io as sio
import scipy.optimize as opt


from checkNNGradients import checkNNGradients
#from nnCostFunctionSinReg import nnCostFunctionSinReg
#from randInitializeWeights import randInitializeWeights
#from nnGradFunctionSinReg import nnGradFunctionSinReg
#from predict import predict


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Setup the parameters you will use for this exercise
    input_layer_size = 400 # 20x20 Input Images of Digits
    hidden_layer_size = 25 # 25 hidden units
    num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

    # Load Training Data
    print("Loading Data ...\n")
    data = sio.loadmat("ex4data1.mat")

    X = data['X']
    y = data['y']
    m = X.shape[0]

    # Load the weights into variables Theta1 and Theta2
    print("Loading Saved Neural Network Parameters ...\n")
    weights = sio.loadmat("ex4weights.mat")
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
    nn_params_ini = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F'))) # Unroll parameters
    print("Shapes: \n\tX: ", X.shape, " - y: ", y.shape, " - theta1: ", theta1.shape, " - theta2: ", theta2.shape, " - params_ini: ", nn_params_ini.shape)


    ## ================ EJ1. Compute Cost (Feedforward) ================
    # To the neural network, you should first start by implementing the
    # forward part of the neural network that returns the cost only.  After
    # implementing the forward propagation to compute the cost, you can verify that
    # your implementation is correct by verifying that you get the same cost
    # as us for the fixed debugging parameters.
    #
    # We suggest implementing the feedforward cost *without* regularization
    # first so that it will be easier for you to debug.
    #
    print("Feedforward Using Neural Network... \n")
    #J = nnCostFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y)
    #print("Cost at parameters (loaded from ex4weights) (this value should be about 0.287629): ", J)

    ## =============== EJ2. Implement Backpropagation ===============
    # You should proceed to implement the backpropagation algorithm
    # for the neural network. You should implement a new function
    # nnGradFunctionSinReg to return the partial
    # derivatives of the parameters.
    #
    # In order to be sure the derivatives are calculated correctly,
    # Use the checkNNGradients(lambda_param) function that creates
    # a small neural network to check the backpropagation gradients
    #
    # Check EB T5 Parte II slides: 19, 20
    print("\nChecking Backpropagation... \n")
    lambda_param = 0
    #checkNNGradients(lambda_param) # Check gradients by running checkNNGradients

    # ================ EJ3. Initializing Parameters ================
    # In this part of the exercise, you will start by
    # implementing a function to initialize the weights of the neural network
    # (randInitializeWeights.py)
    #
    # Check EB T5 Parte II slides: 21, 22
    print("\nInitializating Neural Network Parameters ...\n")
    # initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    # initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    # Unroll parameters (a single column vector)
    # nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

    ## =================== EJ4. Training NN ===================
    # You have now implemented all the code necessary to train a neural
    # network. To train your neural network, we will now use "fmin_cg".
    # Recall that these advanced optimizers are able to train our cost
    # functions efficiently as long as we provide them with the
    # gradient computations.
    #

    print("Training Neural Network ...\n")
    # After you have completed the assignment, change the MaxIter to a larger
    # value to see how more training helps.
    maxiter = 50
    # The cost function needs to be minimized
    #nn_params = opt.fmin_cg(maxiter=maxiter, f=nnCostFunctionSinReg, x0=nn_initial_params, fprime=nnGradFunctionSinReg,
    #                       args=(input_layer_size, hidden_layer_size, num_labels, X, y.flatten()))

    # Obtain theta1 and theta2 back from nn_params
    #theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
    #                        (hidden_layer_size, input_layer_size + 1), 'F')
    #theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1),
    #                        'F')

    ## ================= EJ5. Implement Predict =================
    # After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.
    #pred = predict(theta1, theta2, X, y)
    #print("Training Set Accuracy: ", np.mean(pred == y.flatten()) * 100)
