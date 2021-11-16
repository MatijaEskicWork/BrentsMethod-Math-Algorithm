import random
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

def antennaFunction(n, beta, theta, alpha, deltas):
    sumRange = np.arange(n)
    X, N = np.meshgrid(deltas, sumRange)
    f = np.e ** (-1j * N * (X + beta * alpha * math.cos(theta)))
    return abs(np.sum(f, axis=0))


def plotFunction(n, beta, theta, alpha, numOfPoints, deltaStart, deltaEnd):
    step = deltaEnd - deltaStart / numOfPoints
    x = np.linspace(deltaStart, deltaEnd, numOfPoints)
    plt.plot(x, antennaFunction(n, beta, theta, alpha, x))
    plt.show()


def brentApprox(x1, x2, x3, n, beta, theta, alpha):
    err = float(10**(-7))
    while True:
        matrix = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
        f2 = 0.0
        f1 = 0.0
        f3 = 0.0
        for k in range(n):
            f1 += np.e**(-1j * k * (x1 + beta * alpha * math.cos(theta)))
            f2 += np.e**(-1j * k * (x2 + beta * alpha * math.cos(theta)))
            f3 += np.e**(-1j * k * (x3 + beta * alpha * math.cos(theta)))
        f1 = abs(f1)
        f2 = abs(f2)
        f3 = abs(f3)
        maximum = max(f1, f2, f3)
        arr = np.array([[abs(f1)], [abs(f2)], [abs(f3)]])
        abc = np.dot(np.linalg.inv(matrix),arr)
        minimum = min(f1, f2, f3)
        newX = -1.0 * abc[1][0] / (2.0 * abc[0][0])
        fmax = 0.0
        for k in range(n):
            fmax += np.e**(-1j * k * (newX + beta * alpha * math.cos(theta)))
        fmax = abs(fmax)
        #print("x1={}, x2={}, x3={}, f1={}, f2={}, f3={}, newX = {}".format(x1, x2, x3, f1, f2, f3, newX, fmax))

        if (f1 == minimum):
            x1 = newX
        elif (f2 == minimum):
            x2 = newX
        else:
            x3 = newX

        if (abs(abs(fmax) - abs(maximum)) <= err):
            return newX, fmax


if __name__ == '__main__':
    warnings.simplefilter("ignore", np.ComplexWarning)
    #First part for plotting the function for given parameters
    n = 5
    beta = 20.0 * np.pi
    theta = np.pi / 4.0
    alpha = 1.0 / 20.0
    deltaStart = 0
    deltaEnd = 2.0 * np.pi
    numOfPoints = 100
    plotFunction(n, beta, theta, alpha, numOfPoints, deltaStart, deltaEnd)

    #Second part for calculating maximum
    x1 = random.uniform(3.0, 4.0)
    x2 = random.uniform(3.8, 4.5)
    x3 = 0
    while (x3 < x2):
        x3 = random.uniform(4.0, 5.0)
    print(x1, x2, x3)
    beta, f = brentApprox(x1, x2, x3, n, beta, theta, alpha)
    print("Beta is {} and max value of function is {}".format(beta, f))


