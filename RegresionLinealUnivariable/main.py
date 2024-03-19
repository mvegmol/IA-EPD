# EPD2: Machine Learning - Regresión

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
    plotData(X,y)
    return X, y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('ex1data1.txt')

    ## ======================= EJ2. Función de coste =======================
    print('\nRunning Cost Function ...')
    # Añada una columna con todos sus elementos a 1 a la matriz X como primera columna,
    # e inicializar los parámetros theta a 0



    J_base = computeCost(X, y, theta) # Compute and display initial cost
    print("\tResult EJ2: Cost = ", J_base)

    ## ======================= EJ3. Gradiente =======================
    # Run gradient descent
    print('\nRunning Gradient Descent ...')
    # Some gradient descent settings




    # print theta to screen

    ## ======================= EJ4. Visualización =======================
    # Plot the linear fit


    # Predict values for population sizes of 35, 000 and 70, 000


