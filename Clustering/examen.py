import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def findClosestCentroids(x, initial_centroids):
    m = len(x)
    k = len(initial_centroids)
    idx = np.zeros(m, dtype=int)
    for i in range(m):
        min_dist = float('inf')
        for j in range(k):
            dist = np.linalg.norm(x[i] - initial_centroids[j])
            if dist < min_dist:
                min_dist = dist
                idx[i]=j
    return idx

def computeCentroids(x, idx, k):
    m,n = x.shape
    centroids = np.zeros((k,n))
    for j in range(k):
        ejemplos_asignados = x[idx == j]
        if len(ejemplos_asignados) > 0:
            centroids[j] = np.mean(ejemplos_asignados, axis=0)

    return centroids
def featureNormalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def kMeansInitCentroids(x, k):
    # Número de ejemplos de entrenamiento
    m = x.shape[0]
    # Indices aleatorios
    idx = np.random.randint(0, m, size=k)
    # Seleccionar ejemplos aleatorios como centroides
    centroids = x[idx]
    return centroids

def runKmeans(x, initial_centroids, max_iter, plot=True):
    centroides = initial_centroids
    k = initial_centroids.shape[0]

    for iter in range(max_iter):
        #Los indices mas cercanos
        idx = findClosestCentroids(x,centroides)
        centroides = computeCentroids(x,idx,k)
        if plot:
            plt.scatter(x[:,0], x[:,1], c=idx, cmap='rainbow')
            plt.scatter(centroides[:,0], centroides[:,1], marker='x', s=200,linewidths=3, color='black')
            plt.title(f'Iteration {iter +1}')
            plt.show()

    return centroides, idx

if __name__ == '__main__':

    # Cargar el conjunto de datos, trabajar solo con los atributos sin contar con la clase
    # Normalizar las variables para que posean media 0 y desviación estándar 1, y así asegurar
    # que las variables tengan el mismo peso cuando se realice el clustering
    #Cargamos el datasets de irirs
    iris = datasets.load_iris()

    print(iris)
    #Miramos como se llama la informacion del datasets de irirs y comprobamos que es data
    df = pd.DataFrame(iris['data'])
    df, mu, sigma = featureNormalize(df)
    # Utilizar centroides iniciales obtenidos aleatoriamente de entre los existentes en el
    # conjunto de datos, realizar agrupamiento con k=3 y 10 iteraciones.
    X = df.to_numpy()
    k = 3
    max_iters = 10
    random_initial_centroids = kMeansInitCentroids(X, k)
    centroids, idx = runKmeans(X, random_initial_centroids, max_iters, plot=True)
    print("Centroids computed after ", max_iters, " iterations of K-Means with manual initial centroids:\n", centroids)
    # Dado un ejemplo de flor encontrar el grupo al que pertenece
    # Normalizar el nuevo registro utilizando las medias y desviaciones estándar obtenidas anteriormente
    dato = [8, 4, 4, 4]
    dato_normalizado = (dato - mu) / sigma
    sol = []
    for i in range(k):
        sol.append(np.linalg.norm(dato_normalizado - centroids[i]))
    cluster = np.argmin(sol)
    print("La muestra está más cerca del cluster", cluster)