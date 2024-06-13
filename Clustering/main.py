# EPD5: Machine Learning - Clustering: k-means
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

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

def kMeansInitCentroids(x, k):
    # Número de ejemplos de entrenamiento
    m = x.shape[0]
    # Indices aleatorios
    idx = np.random.randint(0, m, size=k)
    # Seleccionar ejemplos aleatorios como centroides
    centroids = x[idx]
    return centroids


def elbowMethod(X):
    costes = []
    for k in range(1, 10 + 1):
        centroides, idx = runKmeans(X, kMeansInitCentroids(X, k), max_iter=10, plot=False)
        cost = np.sum((X - centroides[idx]) ** 2) / (2 * len(X))
        costes.append(cost)

    # Calcular las diferencias entre los costos
    diff = np.diff(costes)
    # Encontrar el punto de codo utilizando la segunda derivada (diferencia de diferencias)
    acceleration = np.diff(diff)

    # Encontrar el punto de cambio más significativo en la aceleración
    elbow_point = np.argmax(acceleration) + 2  # Sumamos 2 para compensar la segunda derivada

    plt.plot(range(1, 10 + 1), costes, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Cost')
    plt.show()
    return elbow_point


if __name__ == '__main__':
    print("Loading data and setting initial set of centroids\n")
    # Load an example dataset that we will be using
    X = sio.loadmat("ex7data2.mat")['X']
    print(X.shape)
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color="blue")
    plt.show()


    ## ================= EJ1: Find Closest Centroids ====================
    # To help you implement K-Means, we have divided the learning algorithm
    # into two functions -- findClosestCentroids and computeCentroids. In this
    # part, you should implement the code in the findClosestCentroids function.
    #
    # Select an initial set of centroids
    K = 3  # 3 Centroids
    initial_centroids = np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]])

    # Find the closest centroids for the examples using the initial_centroids
    print("Finding closest centroids\n")
    idx = findClosestCentroids(X, initial_centroids)
    print("Closest centroids for the first 3 examples: ", idx[0:3], " (the closest centroids should be 0, 2, 1 respectively)\n")

    ## ===================== EJ2: Compute Means =========================
    # After implementing the closest centroids function, you should now
    # implement the computeCentroids function.
    #
    print("Computing centroids means\n")

    # Compute means based on the closest centroids found in the previous part
    centroids = computeCentroids(X, idx, K)
    print("Centroids computed after initial finding of closest centroids:\n", centroids,
          "\n(the centroids should be [ [2.428301 3.157924] [5.813503 2.633656] [7.119387 3.616684] ] \n")

    ## =================== EJ3: K-Means Clustering ======================
    # After you have completed the two functions computeCentroids and
    # findClosestCentroids, you have all the necessary pieces to run the
    # kMeans algorithm. In this part, you will run the K-Means algorithm on
    # the example dataset we have provided.
    #

    print("Running K-Means clustering one example dataset")
    max_iters = 10

    centroids, idx = runKmeans(X, initial_centroids, max_iters, plot=False)
    print("Centroids computed after ", max_iters, " iterations of K-Means with manual initial centroids:\n", centroids)
    print("\nK-Means done\n")

    ## =================== EJ4: Setting random initial centroids ======================
    # Setting centroids to random examples of the training data and ru-run the kmeans algorithm
    random_initial_centroids = kMeansInitCentroids(X, K)
    centroids, idx = runKmeans(X, random_initial_centroids, max_iters, plot=True)
    print("Centroids computed after ", max_iters, " iterations of K-Means with random initial centroids:\n", centroids)

    ## =================== EJ5: Elbow Method ====================== ##
    elbowMethod(X)

    ## =================== P1: K-Means with SKLEARN ======================
    print("\n\nPROBLEMS\n")

    ## ============= P2: K-Means Clustering on Pixels ===============
    # In this exercise, you will use K-Means to compress an image. To do this,
    # you will first run K-Means on the colors of the pixels in the image and
    # then you will map each pixel on to it's closest centroid.
    #
    print("P2: Running K-Means clustering on pixels from an image")
    img = sio.loadmat('bird_small.mat')['A']

    ## ================= P3: K-Means Clusterin gon pixels using SKLEARN ======================
    # Repeat the clustering using sklearn KMEANS


