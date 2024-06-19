import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.optimize as opt


def training(theta1, theta2, x_train, y_train, input, hidden, num_labels):
    maxiter =2
    nn_initial_param = np.hstack((theta1.ravel(order="F"), theta2.ravel(order="F")))
    nn_params = opt.fmin_cg(maxiter=maxiter, f= nnCostFunction, x0= nn_initial_param, fprime=nnGradFunction, args=(input, hidden, num_labels, x_train, y_train.flatten()))

    theta1 = np.reshape(nn_params[:hidden*(input+1)], (hidden, input+1), order="F")
    theta2 = np.reshape(nn_params[hidden * (input + 1):], (num_labels, hidden + 1), order="F")
    return theta1,theta2

def predict(theta1, theta2, x):
    m = len(x)
    ones = np.ones((ones,x))
    a1 = np.hstack((ones,x))
    a2 = sigmoide(np.dot(a1, theta1.T))
    a2 = np.hstack((ones,a2))
    h = sigmoide(np.dot(a2,theta2.T))
    pred = np.where(h >= 0.5,1,0)
    return pred


def nnGradFunction(params, input,hidden, num_labels, x,y):
    theta1 = np.reshape(params[:hidden*(input+1)], newshape=(hidden,input+1), order='F')
    theta2 = np.reshape(params[hidden*(input+1):], newshape=(num_labels, hidden+1), order='F')

    m = len(y)
    y_d = pd.DataFrame(y)
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    for i in range(x.shape[0]):
        a1,a2,a3 = forward(theta1,theta2,x,i)
        d3 = a3-y_d.iloc[i]#utlima capa
        d2 = np.multiply(np.dot(theta2,d3), np.multiply(a2,1-a2))
        delta1 = delta1 +(np.reshape(d2[1:,],(hidden,1)) @ np.reshape(a1, (1, input+1)))
        delta2 = delta2 +(np.reshape(d3.values,(num_labels,1)) @ np.reshape(a2, (1,hidden+1)))

    delta1 /= m
    delta2 /= m

    gradiente = np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))
    return gradiente

def randInitialWeights(input,output):
    w = np.zeros((output,input+1))
    epsilon_init = 0.12
    w = np.random.rand(output,input+1) * ( 2 * epsilon_init) - epsilon_init
    return w

def optimalHiddenNeurons(input_layer_size, num_labels, X_train, y_train, X_val, y_val):
  # Paso 1: # Inicializamos variables útiles
  num_max_neuronas = 10 # el número máximo de neuronas en el grid
  print('\nCalculando número óptimo de neuronas de la capa oculta... \n')
  print('\nNúmero máximo de neuronas: ',num_max_neuronas)

  arr_accuracy = [] # Inicializamos la lista donde almacenaremos la precisión del conjunto de
  # validación para los diferentes números de neuronas del grid

  # Paso 2: Bucle desde 1 hasta num_max_neuronas (incluido, por eso el +1)
  for hidden_layer_size in range(1, num_max_neuronas+1):
    print('-----\nNúmero de neuronas de la capa oculta: ',hidden_layer_size)

    # Paso 2.1: Inicializar los pesos aleatoriamente con las dimensiones correctas
    initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # Paso 2.2: Entrenamiento de la red neuronal con el número de neuronas de la capa oculta
    theta1_opt, theta2_opt = training(initial_theta1, initial_theta2, X_train, y_train, input_layer_size, hidden_layer_size, num_labels)

    # Paso 2.3: Predicción usando el conjunto de validación
    # parámetros del predict: theta1, theta2, X
    pred = predict(theta1_opt, theta2_opt, X_val)

    # Paso 2.4: Calcular la precisión/accuracy máxima
    arr_accuracy.append(np.mean(pred== y_val )) # Se añade a la lista de precisión
    print("accuracy: ", np.mean(pred== y_val ))

  # Paso 3: Fuera del bucle, encontrar el número de neuronas ocultas con las que se consigue el mejor accuracy
  optimal_hidden_layer_size = np.argmax(arr_accuracy)+1 # +1 porque np.argmax() nos proporciona la posición en la lista del mejor valor. Las posiciones empiezan en 0 y nosotros empezamos en 1 neurona
  print("\n**** El número de neuronas de la capa oculta óptimo es: ", optimal_hidden_layer_size)
  print("**** Con esas neuronas en la capa oculta, el accuracy del conjunto de validación es: ", max(arr_accuracy))

  return optimal_hidden_layer_size
def nnCostFunction(params, input, hidden, num_labels,x,y):
    theta1 = np.reshape(params[:hidden*(input+1)], newshape=(hidden,input+1), order="F")
    theta2 = np.reshape(params[hidden*(input+1):], newshape=(num_labels,hidden),order="F")

    m = len(y)
    suma =0
    y_d = pd.DataFrame(y)

    for i in range(x.shape[0]):
        a1,a2,h = forward(theta1,theta2,x,i)
        #Coste como regresion logistica
        temp1 = y_d.iloc[i] * np.log(h)
        temp2 = (1 - y_d.iloc[i]) * (np.log(1-h))
        temp3 = np.sum(temp1 + temp2)
        suma = suma +temp3
    j = (np.sum(suma) / (-m))
    return j
def forward(theta1,theta2,x,i):
    a1 = np.hstack((1,x[i]))
    a2 = sigmoide(np.dot(theta1,a1))
    a2 = np.hstack((1,a2))
    a3 = sigmoide(np.dot(theta2,a2))
    return a1,a2,a3

#Función sigmoide
def sigmoide(z):
    g = 1 /(1+np.exp(-z))
    return g

if __name__ == "__main__":
    data_train = sio.loadmat("spamTrain.mat")
    x_train = data_train['X']
    y_train = data_train['y']


    ##Datos de validacion
    data_val = sio.loadmat("spamValidation.mat")
    x_val = data_val['X']
    y_val = data_val['y']

    #Datos de test
    data_test = sio.loadmat("spamTest.mat")
    x_test = data_test['X']
    y_test = data_test['y']

    #Añadimos a la x0 a la x
    x_train.insert(0,'x0',1)
    input_layer_size = 1899  # Número de columnas de los datos de entrada (X)
    num_labels = 1  # Spam/No spam
    optimal_hidden_layer_size = optimalHiddenNeurons(input_layer_size, num_labels, x_train, y_train, x_val, y_val)
    initial_theta1 = randInitializeWeights(input_layer_size, optimal_hidden_layer_size)
    initial_theta2 = randInitializeWeights(optimal_hidden_layer_size, num_labels)
    X_train_completo = np.append(x_train, x_val, axis=0)  # IMPORTANTE  axis=0 para que pegue por filas
    y_train_completo = np.append(y_train, y_val, axis=0)  # IMPORTANTE
    theta1_opt, theta2_opt = training(initial_theta1, initial_theta2, x_train_completo, y_train_completo,
                                      input_layer_size, optimal_hidden_layer_size, num_labels)

    # Paso 5: Predecir usando el conjunto de test y calcular el error
    pred = predict(theta1_opt, theta2_opt, X_test)
    print("Accuracy del conjunto de test: ", np.mean(pred == y_test))  # Calcular la precisión/accuracy máxima