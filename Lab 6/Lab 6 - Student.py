import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.io


def featureNormalize(X):
    normalized_X = X
    mu = 0
    sigma = 1

    #TODO: Fill the function featureNormalize(X) 3 lines

    return (normalized_X, mu, sigma)


def pca(X):

    #TODO: Fill the function PCA(X)
    pass


def projectData(X, U, K):
    Z = 0

    #TODO: Fill the function projectData(X,U,K)
    return Z


def recoverData(Z, U, K):
    X_rec = 0
    #TODO: Fill the function recoverData (Z, U, K)
    return X_rec


def displayData(X):
    num_images = len(X)
    rows = int(num_images ** .5)
    cols = int(num_images ** .5)
    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    img_num = 0

    for i in range(rows):
        for j in range(cols):
            # Convert column vector into 32x232 pixel matrix
            #  transpose
            img = X[img_num, :].reshape(32, 32).T
            ax[i][j].imshow(img, cmap='gray')
            img_num += 1

    return (fig, ax)


raw_mat = scipy.io.loadmat("ex7data1.mat")
X = raw_mat.get("X")
plt.cla()
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.show()

X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)

K = 1
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)

plt.cla()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.plot(X_rec[:, 0], X_rec[:, 1], 'rx')
plt.show()

# Loading and Visualizing Face Data
raw_mat = scipy.io.loadmat("ex7faces.mat")
X = raw_mat.get("X")
face_grid, ax = displayData(X[:100, :])
face_grid.show()

X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)

face_grid, ax = displayData(U[:, :36].T)
face_grid.show()

# Dimension Reduction on Faces
K = 100
Z = projectData(X_norm, U, K)

# Visualization of Faces after PCA
K = 100
X_rec = recoverData(Z, U, K)

plt.close()
plt.cla()
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
f, ax1 = displayData(X_norm[:100, :])
f, ax2 = displayData(X_rec[:100, :])
f.show()
print("Done!")