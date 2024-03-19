# EPD2: Machine Learning - Regresión
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sklearn.linear_model


def read_file(file_name):
    # Reading file with data
    print('Loading Data ...', file_name)
    file = pd.read_csv(file_name, names=["poblacion","beneficio"])
    X = pd.DataFrame({'poblacion': file['poblacion']})
    y = pd.DataFrame({'beneficio': file['beneficio']})

    # Plot data: Note: You have to complete the code in plotData.py
    plot_data(X,y)
    return X, y

def plot_data(X, y):
    print("Pintamos la gráfica")
    plt.scatter(X, y , c= "red", marker="x")
    plt.xlabel("Poblacion en ciudad cada 10.000")
    plt.ylabel("Beneficios 10.000 $")
    plt.show()

def compute_cost(x, y, theta):
    #Seguimos la formula de la epd

    m = len(x)
    hipotesis = np.dot(x,theta)
    error  = hipotesis -y
    error_cuadratico = np.power(error, 2)
    sumatorio =np.sum(error_cuadratico, axis=0)
    resultado = sumatorio/2*m
    return resultado

def plot_iteracion_vs_coste(historial_coste):
    iteracion = np.arange(len(historial_coste))
    plt.xlabel("Iteraciones")
    plt.ylabel("Costes")
    plt.plot(iteracion,historial_coste)
    plt.legend()
    plt.show()
def descenso_gradiente(X,y,theta,alpha,iteraciones):
    m = len(X)
    historial_coste = []
    lista = []
    resultado = 0
    # representamos la funcion del descenso gradiente y ademas vamos a ir almacenando los costes con cada uno de los thetas
    for i in range(iteraciones):
        resultado = theta - ((alpha / m) * np.dot(X.T, np.dot(X, theta) - y))
        theta = resultado.copy()
        historial_coste.append(compute_cost(X, y, theta))
        lista.append(theta)
    return historial_coste, resultado, lista
def plot_recta_regresion(x,y,theta_optimo):
    x_values = x['poblacion'].values
    y_values = y['beneficio'].values
    plt.scatter(x_values, y_values, color='red', marker='x', label='Datos de entrenamiento')
    plt.xlabel('Población')
    plt.ylabel('Beneficio')

    # Dibujar la recta de regresión
    y_pred = theta_optimo[0] + theta_optimo[1] * x_values
    plt.plot(x_values, y_pred, color='blue', label='Recta de regresión')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('ex1data1.txt')
    ## ======================= EJ2. Función de coste =======================
    print('\nRunning Cost Function ...')
    # Añada una columna con todos sus elementos a 1 a la matriz X como primera columna,
    # e inicializar los parámetros theta a 0
    #Primero le añadismos la columna de 1 a X
    X['x0'] = 1
    lista_columnas_seleccionar = ['x0', "poblacion"]
    X = X[lista_columnas_seleccionar]

    #Inicializamos theta con las columnas de x y como es univariable sera 1
    theta = np.zeros((X.columns.size, 1))

    coste = compute_cost(X, y, theta) # Compute and display initial cost
    print("\tResult EJ2: Cost = ", coste)

    ## ======================= EJ3. Gradiente =======================
    # Run gradient descent
    print('\nRunning Gradient Descent ...')
    # Some gradient descent settings
    alpha = 0.01
    iteraciones = 1500
    historial_coste, theta_optimo, lista = descenso_gradiente(X,y,theta,alpha,iteraciones)
    #print("Theta optimo:", theta_optimo)
    #print("Lista de thetas:", lista)
    #print("Historial de costes:",historial_coste)

    # print theta to screen
    #Pintamos el historial del coste vs las iteracioens
    plot_iteracion_vs_coste(historial_coste)
    ## ======================= EJ4. Visualización =======================
    # Plot the linear fit

    plot_recta_regresion(X, y, theta_optimo)

    # Predict values for population sizes of 35, 000 and 70, 000
    #Poblacion para hacer las prediccion del beneficio
    #Beneficio = Theta0 + Theta1 * poblacion
    poblacion_1 = 35000
    poblacion_2 = 70000
    #Primera poblacion
    beneficios_pred_1 = theta_optimo[0] + theta_optimo[1] *  poblacion_1
    print("Predicción de beneficio para una población de 35,000:", beneficios_pred_1)
    # Predicción para la segunda población
    beneficio_pred_2 = theta_optimo[0] + theta_optimo[1] * poblacion_2
    print("Predicción de beneficio para una población de 70,000:", beneficio_pred_2)

