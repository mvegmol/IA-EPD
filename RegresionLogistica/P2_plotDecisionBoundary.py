import numpy as np
import matplotlib.pyplot as plt

def mapFeaturePlot(x1,x2,degree):
    """
    take in numpy array of x1 and x2, return all polynomial terms up to the given degree
    """
    out = np.ones(1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j)
            out= np.hstack((out,terms))
    return out

def plotDecisionBoundary(data, theta, X, y, lambda_parameter):

    admitted = data[data['label'] == 1]
    notadmitted = data[data['label'] == 0]

    plt.scatter(admitted['score_1'], admitted['score_2'], label="y=1", c="blue",
                marker="+")
    plt.scatter(notadmitted['score_1'], notadmitted['score_2'], label="y=0",
                c="yellow")

    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")

    u_vals = np.linspace(-1, 1.5, 50)
    v_vals = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u_vals), len(v_vals)))
    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            z[i, j] = mapFeaturePlot(u_vals[i], v_vals[j], 6) @ theta
    plt.contour(u_vals, v_vals, z.T, 0)
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.title("lambda = "+str(lambda_parameter))
    plt.legend()

    plt.show()