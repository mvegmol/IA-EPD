import matplotlib.pyplot as plt

def plotClusters(X,clusters,centroids,ini_centroids):

    # Assigning specific color to each cluster. Assuming 5 for now
    cols={0:'b',1:'g',2:'coral',3:'c',4:'lime'}
    fig,ax=plt.subplots()

    # Plots every cluster points
    for i in range(len(clusters)):
        ax.scatter(X[i][0],X[i][1],color=cols[clusters[i]], marker="+")

    # Plots all the centroids and mark them with a circle around
    for j in range(len(centroids)):
        # Plot current centroids with circle
        ax.scatter(centroids[j][0],centroids[j][1],color=cols[j])
        ax.add_artist(plt.Circle((centroids[j][0], centroids[j][1]), 0.4, linewidth=2, fill=False))
        # Plot initial centroids with ^ and circle in yellow
        ax.scatter(ini_centroids[j][0],ini_centroids[j][1],marker="^",s=150,color=cols[j])
        ax.add_artist(plt.Circle((ini_centroids[j][0], ini_centroids[j][1]), 0.4, linewidth=2, color='y', fill=False))

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("K-means Clustering")
    plt.show()

