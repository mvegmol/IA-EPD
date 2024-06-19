import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
def read_csv(file):
    x = file.drop(columns=['Tipo'])
    y = file['Tipo']

    return x,y


def holdout(x,y, porcetaje):
    x_training = x.sample(round(len(x)*porcetaje))
    y_training = y.iloc[x_training.index]
    x_test = x.drop(x_training.index)
    y_test = y.drop(y_training.index)

    #Reseteamos indices
    x_training = x_training.reset_index(drop=True)
    y_training = y_training.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_training,y_training,x_test,y_test
#Estandarizacion
def normalize(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_norm = X.copy()
    mu = X[numeric_cols].mean()
    sigma = X[numeric_cols].std()
    X_norm[numeric_cols] = (X[numeric_cols] - mu) / sigma
    return X_norm, mu, sigma

def normalize_range(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_norm = X.copy()
    mu = X[numeric_cols].mean()
    max_val = X[numeric_cols].max()
    min_val = X[numeric_cols].min()
    X_norm[numeric_cols] = (X[numeric_cols] - mu) / (max_val - min_val)
    return X_norm, mu, max_val, min_val

def sigmoide(z):
    g = 1 / (1+np.exp(-z))
    return g

def funcion_coste(theta, x, y):
    m = len(y)

    h = sigmoide(np.dot(x,theta))
    j = -(1/m) * np.sum((y * np.log(h))+((1-y)* np.log(1-h)))
    return j

def filter_numeric_columns(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    return X[numeric_cols]

def descenso_gradiente(x, y, theta_inicial, alpha, iteraciones):
    m = len(y)
    coste_historico = []

    theta = theta_inicial.copy()

    for iteracion in range(iteraciones):
        h = sigmoide(np.dot(x, theta))
        error = h - y
        gradiente = (1 / m) * np.dot(x.T, error)

        theta -= alpha * gradiente

        costo = funcion_coste(theta, x, y)
        coste_historico.append(costo)

        if iteracion % 100 == 0:
            print(f"Iteración {iteracion}: Costo {costo}")

    return theta, coste_historico


def predecir(x, theta_optimo):
    # Calculamos la probabilidad de cada clase usando los pesos optimizados
    probabilidad = sigmoide(np.dot(x, theta_optimo))

    # Convertimos las probabilidades a predicciones
    predicciones = np.argmax(probabilidad, axis=1) + 1  # Sumamos 1 porque las clases van de 1 a 6

    return predicciones

def calcular_exactitud(predicciones, y):
    exactitud = np.mean(predicciones == y) * 100
    return exactitud

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Leemos los datos de las estrellas y los separamos
    file = pd.read_csv('caracteristicas_estrellas.csv', names=['Temperature','Luminosity','Radius','Mv','Tipo','color','class'])
    x,y = read_csv(file)
    x_training, y_training, x_test, y_test = holdout(x,y, 0.70)
    #Añadimos a la x que vamos utilizar una filas de unos

    x_training_normalized, mu, max_val, min_val = normalize_range(x_training)
    x_training_normalized.insert(0, 'x0', 1)
    x_training_normalized = filter_numeric_columns(x_training_normalized)
    num_atributos = x_training_normalized.shape[1]
    initial_theta = np.zeros((num_atributos, 1))

    print(f'X {x_training_normalized.shape}')
    print(f'Theta{initial_theta.shape}')

    # Convertimos y a one-hot encoding
    #y_training_onehot = pd.get_dummies(y_training.flatten())  # y es un arrayNumpy // si y es Dataframe habría que hacer y.to_numpy().flatten()
    y_training_onehot = pd.get_dummies(y_training)
    y_training_onehot = y_training_onehot.astype("int")

    # Calculamos el coste
    cost = funcion_coste(initial_theta,x_training_normalized, y_training_onehot)
    print(f"Coste inicial:{cost}")
    # Usamos los datos normalizados y con la columna de unos añadida
    alpha = 1
    iteraciones = 1000
    theta_inicial = np.zeros((num_atributos, y_training_onehot.shape[1]))

    theta_optimo, costo_historico = descenso_gradiente(x_training_normalized.values, y_training_onehot.values,
                                                       theta_inicial, alpha, iteraciones)

    # Graficamos el costo a medida que pasan las iteraciones
    plt.plot(costo_historico)
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Descenso de gradiente: Costo por iteración')
    plt.show()

    print(f"Pesos optimizados:\n{theta_optimo}")

    # Predicciones para el conjunto de entrenamiento
    predicciones_entrenamiento = predecir(x_training_normalized.values, theta_optimo)
    predicciones_entrenamiento = pd.Series(predicciones_entrenamiento)

    # Predicciones para el conjunto de prueba
    # Primero normalizamos x_test
    x_test_normalized = (x_test - mu) / (max_val - min_val)
    x_test_normalized.insert(0, 'x0', 1)
    x_test_normalized = filter_numeric_columns(x_test_normalized)

    # Luego calculamos las predicciones
    predicciones_prueba = predecir(x_test_normalized.values, theta_optimo)
    predicciones_prueba = pd.Series(predicciones_prueba)

    # Exactitud en el conjunto de entrenamiento
    exactitud_entrenamiento = calcular_exactitud(predicciones_entrenamiento, y_training.values)
    print(f'Exactitud en el conjunto de entrenamiento: {exactitud_entrenamiento:.2f}%')

    # Exactitud en el conjunto de prueba
    exactitud_prueba = calcular_exactitud(predicciones_prueba, y_test.values)
    print(f'Exactitud en el conjunto de prueba: {exactitud_prueba:.2f}%')

    '''
    Magnitud 
    import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_csv(file):
    x = file[['Temperature', 'Luminosity', 'Radius']]
    y = file['Mv']
    return x, y

def normalize_range(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_norm = X.copy()
    mu = X[numeric_cols].mean()
    max_val = X[numeric_cols].max()
    min_val = X[numeric_cols].min()
    X_norm[numeric_cols] = (X[numeric_cols] - mu) / (max_val - min_val)
    return X_norm, mu, max_val, min_val

def gradienteRegresionLineal_iterativo(x, y, theta_inicial, alpha, iteraciones):
    m = len(y)
    coste_historico = []

    theta = theta_inicial.copy()

    for iteracion in range(iteraciones):
        h = np.dot(x, theta)
        error = h - y
        gradiente = (1 / m) * np.dot(x.T, error)

        theta -= alpha * gradiente

        costo = np.sum(error ** 2) / (2 * m)
        coste_historico.append(costo)

        if iteracion % 100 == 0:
            print(f"Iteración {iteracion}: Costo {costo}")

    return theta, coste_historico

if __name__ == '__main__':
    # Leemos los datos de las estrellas y los separamos
    file = pd.read_csv('caracteristicas_estrellas.csv', names=['Temperature', 'Luminosity', 'Radius', 'Mv', 'Tipo', 'color', 'class'])
    x, y = read_csv(file)

    # Normalizamos las características de entrada
    x_normalized, mu, max_val, min_val = normalize_range(x)

    # Añadimos una columna de unos para el término de sesgo
    x_normalized.insert(0, 'x0', 1)

    # Inicializamos los parámetros theta
    num_atributos = x_normalized.shape[1]
    initial_theta = np.zeros(num_atributos)

    # Hiperparámetros para el descenso de gradiente
    alpha = 0.01
    iteraciones = 1000

    # Ejecutamos el descenso de gradiente para regresión lineal
    theta_optimo, costo_historico = gradienteRegresionLineal_iterativo(x_normalized.values, y.values, initial_theta, alpha, iteraciones)

    # Calculamos el error cuadrático medio en todo el conjunto de datos
    # Predecimos los valores usando los theta optimizados
    y_pred = np.dot(x_normalized.values, theta_optimo)
    mse = np.mean((y_pred - y.values) ** 2)

    print(f"Error cuadrático medio en todo el conjunto de datos: {mse}")

    # Graficamos el costo a medida que pasan las iteraciones
    plt.plot(costo_historico)
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.title('Descenso de gradiente para regresión lineal: Costo por iteración')
    plt.show()

    print(f"Pesos optimizados:\n{theta_optimo}")
    
    '''