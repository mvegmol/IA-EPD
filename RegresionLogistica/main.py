## Machine Learning Online Class - Exercise 3: Logistic Regression
import pandas as pd
import numpy as np
import scipy.optimize as op


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## ==================== EJ1: Cargar datos y visualizar ====================
    file = pd.read_csv("ex2data1.txt", names=["score_1", "score_2", "label"])
    # X =
    # y =
    # plotData(file)

    ## ==================== EJ2: Hipótesis usando funcion sigmoide ====================
    # Ver EB T4 Diapositivas 27-29
    # sigmoid(z)

    ## ==================== EJ3: Coste y descenso del gradiente con theta inicializados a 0 ====================
    # Función coste debe ser implementada: costFunction(theta, X, y)
    # Función descenso gradiente debe ser implementada: gradientFunction(theta, X, y)
    cost = 0 # Llamar a la funcion coste con theta iniciales
    print("El coste con theta inicializados a cero debe ser aproximadamente 0.693: ", cost)


    ## ==================== EJ4: Optimizador avanzado ====================
    #theta_opt = op.fmin_cg(maxiter=200, f=costFunction, x0=initial_theta.flatten(), fprime=gradientFunction,
    #                         args=(X, y.to_numpy().flatten()))


    print("El coste usando el optimizador avanzado CG debe ser aproximadamente 0.203: ", cost)
    print("Los theta optimos usando el optimizador avanzado CG deben ser aproximadamente: [-25.175949 0.206348 0.20158]: ")#, theta_opt)

    # Frontera de decisión
    # Implementar teniendo en cuenta que nuestros datos son separables linealmente:
    # h = theta0+theta1x1+theta2x2 = 0 --> x2 = - (theta0+theta1x1)/theta2
    #   ejehorizontal = [min(X['score_1']), max(X['score_2'])]
    #   ejevertical = - (theta[0] + np.dot( ejehorizontal, theta[1])) / theta[2]


    ## ==================== EJ5: Predecir candidato admitido o no  ====================
    candidato = np.array([[1, 45, 85]])
    prob = 0 # Implementar función predict
    print("La probabilidad de que el candidato sea admitido debe ser aproximadamente 0.776: ", prob)

    ## ==================== EJ6: Predecir conjunto de entrenamiento X y mostrar exactitud  ====================
    predictions = 0 # Llamar función predict con X completo
    exactitud = 0  # np.mean(predictions == y.to_numpy().flatten())