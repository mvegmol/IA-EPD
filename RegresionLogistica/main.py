

## Machine Learning Online Class - Exercise 3: Logistic Regression
import pandas as pd
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
def plotData(file):
    admitted = file[file['label'] == 1]
    not_admitted = file[file['label'] == 0]
    plt.scatter(x=admitted["score_1"], y=admitted["score_2"], label="Admitido", c="blue", marker="x")
    plt.scatter(x=not_admitted["score_1"], y=not_admitted["score_2"], label="No admitido", c="green", marker="o")
    plt.xlabel("Puntuacion del examen 1")
    plt.ylabel("Puntuacion del examen 2")
    plt.legend()
    plt.show()
def plotDecisionBoundary(data, ejehorizontal, ejevertical):
    # Separamos admitted y no admitted
    admitted = data[data['label'] == 1]
    notadmitted = data[data['label'] == 0]
    # Grafica de cada DataFrame
    plt.scatter(x=admitted['score_1'], y=admitted['score_2'], label="Admitted", c="blue", marker="+")
    plt.scatter(x=notadmitted['score_1'], y=notadmitted['score_2'], label="Not Admitted", c="yellow", marker=".")
    plt.plot(ejehorizontal, ejevertical, label = "Decision boundary") #Lo unico que añadimos nuevo
    plt.xlabel("Examen score 1")
    plt.ylabel("Examen score 2")
    plt.legend()  # Para que me pinte los label de cada scatter
    plt.show()
def sigmoid(z):
    return (1/(1+np.exp(-z)))
def costFunction(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = -(1 / m) * np.sum((y * np.log(h)) + ((1 - y) * np.log(1 - h)), axis=0)
    return J


def gradientFunction_optimization(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * (np.dot(X.T, (h - y)))  # Recuerde: grad debería tener las mismas dimensiones que theta
    return grad

def gradientFunction(theta, x, y):
    m = len(y)
    h = sigmoid(np.dot(x,theta))
    return (1/m)*(np.dot(x.T, (h-y)))


def predict(theta, X):
    # Calcular la hipótesis h = g(X*theta)
    h = sigmoid(np.dot(X, theta))
    return h
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        grad = gradientFunction(theta, X, y)
        theta = theta - alpha * grad
        if i % 100 == 0:  # print cost every 100 iterations
            cost = costFunction(theta, X, y)
            print(f"Iteration {i}: Cost {cost}")
    return theta

# OPCION 2
def holdout(x, y, percentage=0.6):
    # Seleccionamos el numero de filas
    x_training = x.sample(round(porcentage * len(x)))
    # Seleccionamos en y las misma filas que en x
    y_training = y.iloc[x_training.index]
    # En el test de x e y introducideremos las filas que no se encuentran en el training
    x_test = x.drop(x_training.index)
    y_test = y.drop(y_training.index)
    # print("El tamaño del training debe ser: ", round(percentage * len(x)), " - Comprobación: tamaño X_training es ",
    #      len(x_training), " y tamaño y_training es", len(y_training))
    # print("El tamaño del test debe ser: ", len(x) - round(percentage * len(x)), " - Comprobación: tamaño X_test es ",
    #      len(x_test), " y tamaño y_test es", len(y_test))
    # Reseteamos los indices de todos los conjuntos

    x_training = x_training.reset_index(drop=True)
    y_training = y_training.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_training, y_training, x_test, y_test

def normalize_estandarizacion(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## ==================== EJ1: Cargar datos y visualizar ====================
    file = pd.read_csv("ex2data1.txt", names=["score_1", "score_2", "label"])

    x = pd.DataFrame({'score_1':file["score_1"],'score_2':file["score_2"]})
    y = pd.DataFrame({'label':file['label']})
    plotData(file)
    x.insert(0,'ones', 1)
    num_atributos = x.shape[1]  # Si esta operación la hacemos antes de añadir la columna de 1 a X, debemos poner X.shape[1]+1
    initial_theta =  np.zeros((x.columns.size, 1))


    ## ==================== EJ3: Coste y descenso del gradiente con theta inicializados a 0 ====================
    # Función coste debe ser implementada: costFunction(theta, X, y)
    # Función descenso gradiente debe ser implementada: gradientFunction(theta, X, y)
    cost = costFunction(initial_theta,x,y) # Llamar a la funcion coste con theta iniciales
    print("El coste con theta inicializados a cero debe ser aproximadamente 0.693: ", cost)


    ## ==================== EJ4: Optimizador avanzado ====================
    theta_opt = op.fmin_cg(maxiter=200, f=costFunction, x0=initial_theta.flatten(), fprime=gradientFunction,
                              args=(x, y.to_numpy().flatten()))


    print("El coste usando el optimizador avanzado CG debe ser aproximadamente 0.203: ", theta_opt[1])
    print("Los theta optimos usando el optimizador avanzado CG deben ser aproximadamente: [-25.175949 0.206348 0.20158]: ", theta_opt)#, theta_opt)

    # Parámetros de descenso por gradiente
    alpha = 0.01
    num_iters = 200

    # Ejecutar descenso por gradiente
    theta = gradientDescent(x.to_numpy(), y.to_numpy().flatten(), initial_theta, alpha, num_iters)

   

    # Frontera de decisión
    # Implementar teniendo en cuenta que nuestros datos son separables linealmente:
    # h = theta0+theta1x1+theta2x2 = 0 --> x2 = - (theta0+theta1x1)/theta2
    #   ejehorizontal = [min(X['score_1']), max(X['score_2'])]
    #   ejevertical = - (theta[0] + np.dot( ejehorizontal, theta[1])) / theta[2]
    '''
    plotDecisionBoundary(theta_opt, x, y)

    ## ==================== EJ5: Predecir candidato admitido o no  ====================
    candidato = np.array([[1, 45, 85]])
    prob = predict(theta_opt, candidato)
    print("La probabilidad de que el candidato sea admitido debe ser aproximadamente 0.776: ", prob)

    #### División de los datos
    print("Lo mismo pero normalizando y dividiendo los datos con validación cruzada")
    # Normalizar los datos
    x_norm, mu, sigma = normalize(x)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, y_train, x_test, y_test = holdout(x_norm, y)

    # Inicializar theta a cero
    initial_theta = np.zeros(num_atributos)

    # Calcular theta_opt utilizando la función de optimización con los conjuntos de entrenamiento
    theta_opt = op.fmin_cg(maxiter=200, f=costFunction, x0=initial_theta.flatten(), fprime=gradientFunction,
                           args=(x_train, y_train.to_numpy().flatten()))

    # Predecir y calcular la exactitud utilizando theta_opt y los conjuntos de prueba
    predictions = predict(theta_opt, x) # Llamar función predict con X completo
    exactitud = np.mean(predictions == y.to_numpy().flatten())  # Calcular la exactitud
    print("La exactitud del modelo en el conjunto de entrenamiento es: ", exactitud)

    ## ==================== EJ6: Predecir conjunto de entrenamiento X y mostrar exactitud  ====================
    predictions = 0 # Llamar función predict con X completo
    exactitud = 0  # np.mean(predictions == y.to_numpy().flatten())
'''