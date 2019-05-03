import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import csv
import math


def read_data():
    data = []
    # TODO: Read the file 'data1.csv' into the variable data.
    # data contains the training data together with labelled classes.
    return data


def multivariate_normal_gaussian(x, mu, sigma):
    prob = 0
    # TODO: Implement the multivariate normal gaussian distribution with parameters mu and sigma.
    return prob


data = read_data()

# TODO: Estimate the parameters of the Gaussian distributions of the given classes.


colors = ['r', 'g', 'b', 'c', 'y']
# TODO: Do a scatter plot for the data, where each class is coloured by the colour corresponding
# TODO: to its index in the colors array.
# Class 1 should be coloured in red, Class 2 should be coloured in green and Class 3 should be coloured in blue.

def find_decision_boundary():
    w0 = 0
    w = 0
    #TODO: Find the coefficients of the decision boundary. Pass the required parameters to the function.
    return w, w0

# TODO: Generate a 3D-plot for the generated distributions. x-axis and y-axis represent the features of the data, where
# TODO: z-axis represent the Gaussian probability at this point.
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
Zplane,_ = np.meshgrid(np.linspace(0, 50, 300), y)

for i in range (Z.shape[0]):
    for j in range (Z.shape[1]):
        # TODO: Fill in the matrix Z which will represent the probability distribution of every point.
        Z[i,j] = 0

#TODO: Call find_decision_boundary(..) to find the coefficients of the plane in the form W.T@X + W0 = 0
Xplane = 0
Yplane = 0

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, zorder=0.3)
ax.plot_surface(Xplane, Yplane, Zplane, cmap='plasma', linewidth=0, antialiased=True, zorder=0.8, alpha=0.9)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()