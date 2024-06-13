# EPD6: Machine Learning - Recommender systems

import scipy.io as sio
import numpy as np
import scipy.optimize as opt


def cofiCostFuncSinReg(params, Y, R, atributos):
    peliculas = Y.shape[0]
    usuarios = Y.shape[1]
    #desenrrollamos la x y theta
    x  = np.reshape(params[0:peliculas*atributos], newshape=(peliculas,atributos),
                    order="F")
    theta = np.reshape(params[peliculas*atributos:], newshape=(atributos,usuarios),
                       order="F")
    #una vez tenemos los parametros pasamos a calcular el coste
    error_1 = np.dot(x,theta)-Y
    error_2 = np.multiply(error_1,R) #multiplico por R para que los que no la hayan calificado de 0
    error_3 = np.power(error_2,2)
    error_4 = (1/2)*np.sum(error_3)
    return error_4

def cofiGradientFuncSinReg(params, Y, R, atributos):
    # Obtener las dimensiones de las matrices Y y R
    peliculas = Y.shape[0]
    usuarios = Y.shape[1]

    # Reshape de los parámetros en las matrices X y Theta
    x = np.reshape(params[0:peliculas * atributos], newshape=(peliculas, atributos),
                   order="F")

    theta = np.reshape(params[peliculas * atributos:], newshape=(atributos, usuarios),
                       order="F")
    # Calcular el error entre la predicción y las calificaciones reales
    error = np.dot(x, theta) - Y
    # Multiplicar el error por la máscara R para considerar solo las calificaciones existentes
    error_calificadas = np.multiply(error, R)
    # Calcular los gradientes respecto a Theta y X
    theta_grad = np.dot(x.T, error_calificadas)  # Gradiente respecto a Theta
    x_grad = np.dot(error_calificadas, theta.T)  # Gradiente respecto a X
    # Concatenar y aplanar los gradientes para devolver un vector unidimensional
    grad = np.hstack((np.ravel(x_grad, order="F"), np.ravel(theta_grad, order="F")))
    return grad

def cofiCostFuncReg(params, Y, R, atributos, lambda_param):
    peliculas = Y.shape[0]
    usuarios = Y.shape[1]
    #desenrollamos x y theta
    x = np.reshape(params[0:peliculas*atributos],newshape=(peliculas,atributos),order="F")
    theta = np.reshape(params[peliculas*atributos:],newshape=(atributos,usuarios),order="F")

    hipotesis = np.dot(x,theta) -Y
    coste_parte1 = np.sum(np.power(np.multiply(hipotesis,R),2))

    coste_parte2 = lambda_param * np.sum(np.power(theta,2))
    coste_parte3 = lambda_param* np.sum(np.power(x,2))

    coste_final = (1/2)*(coste_parte1+coste_parte2+coste_parte3)

    return coste_final

def cofiGradientFuncReg(params, Y, R, atributos, lambda_param):
    peliculas = Y.shape[0]
    usuarios = Y.shape[1]
    #desenrollamos x y theta
    x = np.reshape(params[0:peliculas*atributos],newshape=(peliculas,atributos),order="F")
    theta = np.reshape(params[peliculas*atributos:],newshape=(atributos,usuarios),order="F")
    #calculamos el gradiente
    gradi_1parte = np.multiply((np.dot(x,theta)-Y),R)
    #calculamos el gradiente de theta
    theta_gradiente = np.dot(x.T,gradi_1parte)+(lambda_param*theta)
    #calculamos el gradiente de x
    x_gradiente = np.dot(gradi_1parte,theta.T)+(lambda_param*x)
    #enrollamos el gradiente final de x y theta
    gradiente_final = np.hstack((np.ravel(x_gradiente,order="F"),np.ravel(theta_gradiente,order="F")))

    return gradiente_final

def descenso_gradiente(Y, R, X, Theta, atributos, lambda_param, alpha, num_iter):
    peliculas = Y.shape[0]
    usuarios = Y.shape[1]

    mejor_coste = float('inf')  # Inicializar el mejor costo con infinito

    for iteracion in range(num_iter):
        # Calcular el error entre la predicción y las calificaciones reales
        error = np.dot(X, Theta) - Y

        # Multiplicar el error por la máscara R para considerar solo las calificaciones existentes
        error_calificadas = np.multiply(error, R)

        # Calcular los gradientes respecto a Theta y X con regularización
        theta_grad = np.dot(X.T, error_calificadas) + lambda_param * Theta
        x_grad = np.dot(error_calificadas, Theta.T) + lambda_param * X

        # Actualizar los parámetros con el descenso de gradiente
        Theta =Theta- alpha * theta_grad
        X = X- alpha * x_grad

        # Calcular el coste con la nueva Theta y X
        coste = cofiCostFuncReg(np.hstack((np.ravel(X, order="F"), np.ravel(Theta, order="F"))), Y, R, atributos,
                                lambda_param)

        print(f"Iteración {iteracion + 1}, Coste: {coste}")

        # Actualizar el mejor conjunto de parámetros si encontramos un costo menor
        if coste < mejor_coste:
            mejor_coste = coste
            mejor_X = np.copy(X)
            mejor_Theta = np.copy(Theta)

    return mejor_X, mejor_Theta


if __name__ == '__main__':

    # =============== EJ1: Cargar datos ================
    print('Loading movie ratings dataset.')

    #Leemos el fichero
    movies = sio.loadmat("ex8_movies.mat")
    Y = movies['Y'] # [n_items, n_users] puntuaciones de 1-5
    R = movies['R'] # [n_items, n_users] R(i,j)=1 si usuario j puntuó pelicula i

    #Calculamos la media
    print('\tAverage rating for the first movie (Toy Story): ', Y[0, np.where(R[0, :] == 1)[0]].mean(), "/5\n")

    #  Cargar parámetros preentrenados (X, Theta, num_users, num_movies, num_features)
    params_data = sio.loadmat('ex8_movieParams.mat')
    X = params_data['X']
    Theta = params_data['Theta']
    Theta = Theta.T
    # ...
    print("Shape de X: ", X.shape)  # [n_items, features]
    print("Shape de Theta: ", Theta.shape)  # [features, n_users]

    # ============ EJ2: Función Coste sin Regularización ===========
    #  Filtrado colaborativo de sistemas de recomendación

    ### Subconjunto de datos para que ejecute más rápidamente
    users = 4
    movies = 5
    features = 3

    X_sub = X[0:movies,0:features]
    Theta_sub = Theta[0:features,0:users]
    Y_sub = Y[0:movies,0:users]
    R_sub = R[0:movies,0:users]

    #params = ... # Desenrollar: primero X_sub luego Theta_sub
    params = np.hstack((np.ravel(X_sub,order="F"), np.ravel(Theta_sub, order='F')))
    J = cofiCostFuncSinReg(params, Y_sub, R_sub, features)
    print("Cost without regularization at loaded parameters: ", J, "(this value should be about 22.22)")

    # ============ EJ3: Gradiente sin Regularización ===========
    # Filtrado colaborativo de sistemas de recomendación
    grad = cofiGradientFuncSinReg(params, Y_sub, R_sub, features)
    #
    print("Gradient without regularization at loaded parameters: \n", grad)
    lambda_param = 0
    #checkNNGradients(lambda_param)

    # ========= EJ4: Función coste con Regularización ========
    # Filtrado colaborativo de sistemas de recomendación
    lambda_param = 1.5
    J = cofiCostFuncReg(params, Y_sub, R_sub, features, lambda_param)
    print("\n\nCost with regularization at loaded parameters: ", J, "(this value should be about 31.34)")

    grad = cofiGradientFuncReg(params, Y_sub, R_sub, features, lambda_param)
    print("Gradient with regularization at loaded parameters: \n", grad)
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
    params = np.hstack((np.ravel(X,order="F"),np.ravel(Theta,order="F")))

    # Algoritmo de optimización
    fmin_1 = opt.fmin_cg(maxiter=maxiter, f=cofiCostFuncReg, x0=params, fprime=cofiGradientFuncReg,
                         args=(Y, R, features, lambda_param))

    X_fmin = np.reshape(fmin_1[0:movies * features], newshape=(movies, features), order="F")
    Theta_fmin = np.reshape(fmin_1[movies * features:], newshape=(features, users), order="F")

    x_mejor, theta_mejor = descenso_gradiente(Y, R, X, Theta, features, lambda_param, 0.0001, maxiter)
    print("Theta Mejkor", theta_mejor)
    print("X mejor", x_mejor)
    predicciones_mejor = np.dot(x_mejor, theta_mejor)


    # ============== EJ6: Predicciones ===============
    #predictions =
    predicciones = np.dot(X_fmin, Theta_fmin)
    # Solo el usuario j
    j = 2
    res_user = np.zeros((movies, 1))
    res_user_mejor = np.zeros((movies, 1))
    pred_userj = predicciones[:, j]  # Seleccionar el usuario j
    # Para cada película: A las que tenían valor previo le ponemos un 0 y a las que hemos predicho el valor de su predicción
    for i in range(movies):
        res_user[i, 0] = np.where(R[i, j] == 0,
                                  predicciones[i, j], 0)
        # Pongo 0 si no se cumple la condicion y la prediccion en caso contrario
    for i in range(movies):
        res_user_mejor[i, 0] = np.where(R[i, j] == 0,
                                        predicciones_mejor[i, j], 0)
    idx = np.argsort(res_user, axis=0)[
          ::-1]  # Ordenar por las predicciones de menor a mayor y coger sus índice. [::-1] significa que le damos la vuelta a la salida: es decir lo colocamos de mayor a menor

    # Leer el fichero con los nombres de cada película
    movie_idx = {}
    f = open('movie_ids.txt', encoding='ISO-8859-1')
    for line in f:
        tokens = line.split(' ')
        tokens[-1] = tokens[-1][:-1]
        movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

    print("Top 10 movie predictions:")
    for i in range(10):
        j = int(idx[i])
        print('Predicted rating of {0} for movie {1}.'.format(str(float(res_user[j])), movie_idx[j]))
        print('Predicted mejor rating of {0} for movie {1}.'.format(str(float(res_user_mejor[j])), movie_idx[j]))