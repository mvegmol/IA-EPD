import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.optimize as opt


#from checkNNGradients import checkNNGradients
#from nnCostFunctionSinReg import nnCostFunctionSinReg
#from randInitializeWeights import randInitializeWeights
#from nnGradFunctionSinReg import nnGradFunctionSinReg
#from predict import predict
def nnGradFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y):
    # Paso 1: Saco theta1 y theta2 de nn_params_ini
    theta1 = np.reshape(a=nn_params_ini[:hidden_layer_size * (input_layer_size + 1)],
                        newshape=(hidden_layer_size, input_layer_size + 1),
                        order="F")

    theta2 = np.reshape(a=nn_params_ini[hidden_layer_size * (input_layer_size + 1):],
                        newshape=(num_labels, hidden_layer_size + 1),
                        order="F")

    # Paso 2: One hot encoding
    y_d = pd.get_dummies(y.flatten())  # y es un arrayNumpy // si y es Dataframe habría que hacer y.to_numpy().flatten()
    # Paso 3: Para cada fila de X
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for i in range(len(X)):
        # Paso 3.1: Llamo al forward para sacar a1, a2 y a3(h)
        a1, a2, a3 = forward(theta1, theta2, X, i)
        # Paso 3.2: Calculo los errores (en capa 1 no hay)
        d3 = a3 - y_d.iloc[i]
        d2_parte1 = np.dot(theta2.T, d3)
        d2_parte2 = np.multiply(a2, 1 - a2)
        d2 = np.multiply(d2_parte1, d2_parte2)
        # Paso 3.3: Acumulo el error en delta1 y delta2
        delta1_parte1 = np.reshape(a = d2[1:], newshape = (hidden_layer_size, 1)) # Quito la bias
        delta1_parte2 = np.reshape(a = a1, newshape = (1, input_layer_size + 1))
        delta1 = delta1 + np.dot(delta1_parte1, delta1_parte2)
        delta2_parte1 = np.reshape(a = d3.to_numpy() #d3 es una Serie. Para aplicar el np.reshape (de numpy) tengo que convertirlo a numpy/array
                                        ,newshape=(num_labels, 1))
        delta2_parte2 = np.reshape(a = a2, newshape = (1, hidden_layer_size + 1))
        delta2 = delta2 + np.dot(delta2_parte1, delta2_parte2)
    m = len(X)
    delta1 = delta1 / m
    delta2 = delta2 / m
    return np.hstack((delta1.ravel(order = "F"), delta2.ravel(order = "F")))

def nnCostFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y):
    #Sacamos las thetas 1 t theta 2 del nn_params_ini
    theta1 = np.reshape(a=nn_params_ini[:hidden_layer_size *(input_layer_size+1)],
                        newshape=(hidden_layer_size, input_layer_size+1),
                        order="F")
    theta2 = np.reshape(a= nn_params_ini[hidden_layer_size *(input_layer_size+1):],
                        newshape=(num_labels, hidden_layer_size+1),
                        order="F")
    #Paso 2 : One hot coding
    y_d = pd.get_dummies(y.flatten())  # y es un arrayNumpy // si y es Dataframe habría que hacer y.to_numpy().flatten()
    y_d = y_d.astype("int")
    suma=0
    #Para cada fila X
    for i in range(len(X)):
        # Paso 3.1: Forward // Feetforward
        a1,a2, h = forward(theta1,theta2,X,i)
        # Paso 3.2: Coste como regresión logística
        y_1 = y_d.iloc[i] * np.log(h)
        y_0 = (1 - y_d.iloc[i]) * (np.log(1 - h))
        temp = np.sum(y_1 + y_0)
        suma = suma + temp

    m = len(X)
    J = -(1 / m) * suma
    return J
def forward(theta1, theta2, X, i): # i es la fila
    a1 = np.hstack((1, X[i])) # Le pongo la bias
    a2 = sigmoide(np.dot(theta1, a1))
    a2 = np.hstack((1, a2)) # Le pongo la bias
    h = sigmoide(np.dot(theta2, a2)) # No se pone la bias con la última capa
    return a1, a2, h


def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def randInitializeWeights(layer_input, layer_output): #Forward (me quedo con h = a3)
    epsilon = 0.12
    return np.random.rand(layer_output, layer_input + 1) * (2 * epsilon) - epsilon

def predict(theta1, theta2, X):
    ones = np.ones((len(X), 1))
    a1 = np.hstack((ones, X))
    a2 = sigmoide(np.dot(a1, theta1.T))
    a2 = np.hstack((ones, a2))
    h = sigmoide(np.dot(a2, theta2.T))
    return np.argmax(h, axis = 1) + 1

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
    print("Feedforward Using Neural Network... \n")
    J = nnCostFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y)
    print("Cost at parameters (loaded from ex4weights) (this value should be about 0.287629): ", J)

    ## =============== EJ2. Implement Backpropagation ===============
    print("\nChecking Backpropagation... \n")
    lambda_param = 0


    # ================ EJ3. Initializing Parameters ================
    # In this part of the exercise, you will start by
    # implementing a function to initialize the weights of the neural network
    # (randInitializeWeights.py)
    #
    # Check EB T5 Parte II slides: 21, 22
    print("\nInitializating Neural Network Parameters ...\n")
    initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    # Unroll parameters (a single column vector)
    nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

    ## =================== EJ4. Training NN ===================
    print("Training Neural Network ...\n")
    # After you have completed the assignment, change the MaxIter to a larger
    # value to see how more training helps.
    maxiter = 50
    # The cost function needs to be minimized
    nn_params = opt.fmin_cg(maxiter=maxiter, f=nnCostFunctionSinReg, x0=nn_initial_params, fprime=nnGradFunctionSinReg,
                           args=(input_layer_size, hidden_layer_size, num_labels, X, y.flatten()))

    # Obtain theta1 and theta2 back from nn_params
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                            (hidden_layer_size, input_layer_size + 1), 'F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1),
                           'F')

    ## ================= EJ5. Implement Predict =================
    # After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.
    #pred = predict(theta1, theta2, X, y)
    #print("Training Set Accuracy: ", np.mean(pred == y.flatten()) * 100)
