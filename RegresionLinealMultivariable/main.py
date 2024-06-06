import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sklearn.linear_model

def normalize_features(x):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost_multi(X, y, theta):
    """
    Calcula el costo para la regresión lineal con múltiples variables.

    Args:
    X : array_like
        Matriz de características (m x n).
    y : array_like
        Vector de valores objetivos (m x 1).
    theta : array_like
        Vector de parámetros (n x 1).

    Returns:
    J : float
        El costo de usar theta como los parámetros de la regresión.
    """
    m = len(y)
    J = (1 / (2 * m)) * np.sum(np.square(X.dot(theta) - y))
    return J


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    """
    Realiza el descenso por gradiente para aprender theta.

    Args:
    X : array_like
        Matriz de características (m x n).
    y : array_like
        Vector de valores objetivos (m x 1).
    theta : array_like
        Vector de parámetros (n x 1).
    alpha : float
        Tasa de aprendizaje.
    num_iters : int
        Número de iteraciones.

    Returns:
    theta : array_like
        Vector de parámetros optimizados.
    J_history : array_like
        Historico del costo en cada iteración.
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
        J_history[i] = compute_cost_multi(X, y, theta)

    return theta, J_history

if __name__ == '__main__':

    data = pd.read_csv('ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print(X)
    print(y)

    X_norm, mu, sigma = normalize_features(X)
    X_norm = np.concatenate([np.ones((X_norm.shape[0], 1)), X_norm], axis=1)
    # Inicializar parámetros y configuraciones para el descenso por gradiente
    alpha = 0.03
    num_iters = 200
    theta = np.zeros(X_norm.shape[1])

    # Ejecutar el descenso por gradiente
    theta, J_history = gradient_descent_multi(X_norm, y, theta, alpha, num_iters)
    plt.plot(range(1, num_iters + 1), J_history, 'b-')
    plt.xlabel('Número de iteraciones')
    plt.ylabel('Costo J')
    plt.title('Convergencia del descenso por gradiente')
    plt.show()
    # Normalizar las características de la casa de prueba
    house = np.array([1650, 3])
    house_norm = (house - mu) / sigma
    house_norm = np.concatenate([[1], house_norm])

    # Predicción de precio
    price_pred = house_norm.dot(theta)
    print(f"El precio predicho para una casa de 1650 pies cuadrados y 3 habitaciones es: ${price_pred:.2f}")