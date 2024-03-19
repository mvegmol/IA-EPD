# EPD6: Machine Learning - Recommender systems

import scipy.io as sio
import numpy as np
import scipy.optimize as opt

# Press the green button in the gutter to run the script.
#from cofiCostFuncReg import cofiCostFuncReg
#from cofiCostFuncSinReg import cofiCostFuncSinReg
#from cofiGradientFuncReg import cofiGradientFuncReg
#from cofiGradientFuncSinReg import cofiGradientFuncSinReg
#from checkNNGradients import checkNNGradients

if __name__ == '__main__':

    # =============== EJ1: Cargar datos ================
    print('Loading movie ratings dataset.')
    #
    movies = sio.loadmat("ex8_movies.mat")
    Y = movies['Y'] # [n_items, n_users] puntuaciones de 1-5
    R = movies['R'] # [n_items, n_users] R(i,j)=1 si usuario j puntuó pelicula i

    print('\tAverage rating for the first movie (Toy Story): ', Y[0, np.where(R[0, :] == 1)[0]].mean(), "/5\n")

    #  Cargar parámetros preentrenados (X, Theta, num_users, num_movies, num_features)
    params_data = sio.loadmat('ex8_movieParams.mat')
    X = params_data['X']
    Theta = params_data['Theta']
    # ...
    print("Shape de X: ", X.shape)  # [n_items, features]
    print("Shape de Theta: ", Theta.shape)  # [features, n_users]

    # ============ EJ2: Función Coste sin Regularización ===========
    #  Filtrado colaborativo de sistemas de recomendación

    ### Subconjunto de datos para que ejecute más rápidamente
    users = 4
    movies = 5
    features = 3

    #X_sub = ...
    #Theta_sub = ...
    #Y_sub = ...
    #R_sub = ...

    #params = ... # Desenrollar: primero X_sub luego Theta_sub

    #J = cofiCostFuncSinReg(params, Y_sub, R_sub, features)
    #print("Cost without regularization at loaded parameters: ", J, "(this value should be about 22.22)")

    # ============ EJ3: Gradiente sin Regularización ===========
    # Filtrado colaborativo de sistemas de recomendación
    #grad = cofiGradientFuncSinReg(params, Y_sub, R_sub, features)
    #print("Gradient without regularization at loaded parameters: \n", grad)
    #lambda_param = 0
    #checkNNGradients(lambda_param)

    # ========= EJ4: Función coste con Regularización ========
    # Filtrado colaborativo de sistemas de recomendación
    lambda_param = 1.5
    #J = cofiCostFuncReg(params, Y_sub, R_sub, features, lambda_param)
    #print("\n\nCost with regularization at loaded parameters: ", J, "(this value should be about 31.34)")

    #grad = cofiGradientFuncReg(params, Y_sub, R_sub, features, lambda_param)
    #print("Gradient with regularization at loaded parameters: \n", grad)
    #checkNNGradients(lambda_param)

    # ============== EJ5: Inicialización random de X y Theta. Algoritmo de optimización con regularización. ===============
    # Valores del conjunto de datos total
    movies = Y.shape[0]  # 1682
    users = Y.shape[1]  # 944
    features = 10
    lambda_param = 1.5
    maxiter = 200

    # Inicialización de X y Theta
    X = np.random.rand(movies, features) * (2*0.12)
    Theta = np.random.rand(features, users) * (2*0.12)
    #params = # Desenrollar: primero X luego Theta

    # Algoritmo de optimización
    #fmin_1 = opt.fmin_cg(maxiter=maxiter, f=cofiCostFuncReg, x0=params, fprime=cofiGradientFuncReg,
    #                  args=(Y, R, features, lambda_param))

    # Enrollar el resultado
    #X_fmin =
    #Theta_fmin =


    # ============== EJ6: Predicciones ===============
    #predictions =

    # Solo el usuario j
    j = 2
    res_user = np.zeros((movies, 1))
    #pred_userj = predictions... # Seleccionar el usuario j
    # Para cada película: A las que tenían valor previo le ponemos un 0 y a las que hemos predicho el valor de su predicción

    idx = np.argsort(res_user, axis=0)[::-1] # Ordenar por las predicciones de menor a mayor y coger sus índice. [::-1] significa que le damos la vuelta a la salida: es decir lo colocamos de mayor a menor

    # Leer el fichero con los nombres de cada película
    movie_idx = {}
    f = open('movie_ids.txt',encoding = 'ISO-8859-1')
    for line in f:
        tokens = line.split(' ')
        tokens[-1] = tokens[-1][:-1]
        movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

    print("Top 10 movie predictions:")
    for i in range(10):
        j = int(idx[i])
        print('Predicted rating of {0} for movie {1}.'.format(str(float(res_user[j])), movie_idx[j]))
