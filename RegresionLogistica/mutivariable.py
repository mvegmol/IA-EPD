import numpy as np
import scipy.io as sio
import scipy.optimize as op
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
def leer_fichero(file):
    data = sio.loadmat(file)
    x = pd.DataFrame(data['X'])
    y = pd.DataFrame(data['y'])
    m = x.shape[0]

    print("Cargando los datos ...")
    print("El tamaño de X: ", m)
    print("La longitud del vector y es: ", len(y))

    return x,y,m

def sigmoide(z):
    g = 1 / (1+np.exp(-z))
    return g

def funcion_coste(theta, x, y):
    m = len(y)
    h = sigmoide(np.dot(x,theta))
    j = -(1/m) * np.sum((y * np.log(h))+((1-y)* np.log(1-h)))
    return j

def funcion_gradiente_optimizador(theta, x, y):
    m = len(y)#Numero de atributos
    hipotesis = sigmoide(np.dot(x,theta)) #La hipotesiss es el sigmaoide por x*theta
    gradiente = (1/m)* (np.dot(x.T,(hipotesis-y)))
    return gradiente
def cambiar_y(y, clase):
    return pd.get_dummies(y.to_numpy().flatten())[clase]

def training(theta_incial, x_train, y_train, num_clases):
    #Lista vacias para todas las thetas,costes, y clases
    all_theta =[]
    all_coste= []
    all_class = []
    #Recorremos todas las funciones
    for current_class in range(1, num_clases+1):
        #Funcion de descenso de gracddientes con el gradiente optimizador
        res_optimization = op.fmin_cg(maxiter=2, f=funcion_coste, x0=theta_incial.flatten(),
                                      fprime=funcion_gradiente_optimizador,  # Derivada
                                      args=( x_train, cambiar_y(y_train, current_class).to_numpy().flatten()),
                                      full_output=True)  # Indicando full_output=True la salida del algoritmo de optimización
        # será en la posición [0] los theta óptimos y en la posición [1] el coste alcanzado por esos theta optimos
        all_theta.append(res_optimization[0])  # Theta óptimos del algoritmo de optimización
        all_coste.append(res_optimization[1])  # Coste del algoritmo de optimización
        all_class.append(current_class)  # Clase actual

    #Devolvemos todas las clases,thetasy coste en un mismo dataFRAME
    df_opt = pd.DataFrame({'class':all_class, 'theta':all_theta, 'cost':all_coste})
    return df_opt

def holdout(x,y,porcentage):
    # Seleccionamos el numero de filas
    x_training = x.sample(round(porcentage*len(x)))
    #Seleccionamos en y las misma filas que en x
    y_training = y.iloc[x_training.index]
    #En el test de x e y introducideremos las filas que no se encuentran en el training
    x_test = x.drop(x_training.index)
    y_test = y.drop(y_training.index)
    #print("El tamaño del training debe ser: ", round(percentage * len(x)), " - Comprobación: tamaño X_training es ",
    #      len(x_training), " y tamaño y_training es", len(y_training))
    #print("El tamaño del test debe ser: ", len(x) - round(percentage * len(x)), " - Comprobación: tamaño X_test es ",
    #      len(x_test), " y tamaño y_test es", len(y_test))
    #Reseteamos los indices de todos los conjuntos

    x_training = x_training.reset_index(drop=True)
    y_training = y_training.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop = True)

    return x_training, y_training, x_test, y_test

def predecir(res_opt, x):
    m = len(x)
    lista_instancias = []
    lista_predicciones = []
    for i in range( len(x)):
        lista_h =[]
        for clase_actual in res_opt['class']:
            h = sigmoide(np.dot(x.iloc[i], res_opt['theta'][clase_actual-1]))
            lista_h.append(h)
        pred = np.argmax(lista_h)+1
        lista_predicciones.append(pred)
        lista_instancias.append(i)
    return pd.DataFrame({'instance': lista_instancias, 'prediction':lista_predicciones})


def accuracy_model(y_real, y_pre):
    return np.mean(y_pre['prediction']==y_real[0])



if __name__ == '__main__':
    # Para ignorar las advertencias
    np.seterr(all='ignore')

    #Leemos el archivo
    x,y,m = leer_fichero("ex4data1.mat")

    #Añadimos una columna de 1 a la primera fila de x
    x.insert(0,'x0',1)

    #Creamos theta inicial
    num_atributos  = x.shape[1]
    #Creamos la theta Inicial
    theta_inicial = np.zeros((num_atributos,1),dtype=np.float64)

    num_clases = 10
    #Hold-out
    x_training, y_training, x_test, y_test = holdout(x,y,0.7)

    #Entrenamiento
    res_opt_training = training(theta_inicial,x_training,y_training,num_clases)

    #Prediccion
    res_pred_training = predecir(res_opt_training,x_training)
    accuracy_training = accuracy_model(y_training, res_pred_training)
    print("Media de entrenamiento: ", accuracy_training)
    print("Media entrenamiento sklearn: ", metrics.accuracy_score(y_training,res_pred_training['prediction']))


    #Prediccion del test
    res_pred_test = predecir(res_opt_training,x_test)
    accuracy_test = accuracy_model(y_test, res_pred_test)
    print("Media de test: ", accuracy_test)
    print("Media test sklearn: ", metrics.accuracy_score(y_test, res_pred_test['prediction']))
