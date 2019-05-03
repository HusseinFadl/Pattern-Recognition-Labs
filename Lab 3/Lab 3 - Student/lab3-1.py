import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import math


def read_data():
    data = []
    # TODO: Read the file 'data1.csv' into the variable data.
    # data contains the training data together with labelled classes.
    return data


def read_test_data():
    test_data = []
    test_data_true = []
    # TODO: Read the file 'test_data.csv' and 'test_data_true.csv' into the variables test_data and test_data_true.
    # test_data contains the unlabelled test class.
    # test_data_true contains the actual classes of the test instances, which you will compare
    # against your predicted classes.
    return test_data, test_data_true


def multivariate_normal_gaussian(x, mu, sigma):
    prob = 0
    # TODO: Implement the multivariate normal gaussian distribution with parameters mu and sigma.
    return prob


training_data = read_data()
test_data, test_data_true = read_test_data()

# TODO: Estimate the parameters of the Gaussian distributions of the given classes.


colors = ['r', 'g', 'b', 'c', 'y']
# TODO: Do a scatter plot for the data, where each class is coloured by the colour corresponding
# TODO: to its index in the colors array.
# Class 1 should be coloured in red, Class 2 should be coloured in green and Class 3 should be coloured in blue.


# TODO: Apply the Bayesian Classifier to predict the classes of the test points.
predicted_classes = []

# TODO: Compute the accuracy of the generated Bayesian classifier.
accuracy = 0
print('Accuracy = ' + str(accuracy*100) + '%')


# TODO: Generate a 3D-plot for the generated distributions. x-axis and y-axis represent the features of the data, where
# TODO: z-axis represent the Gaussian probability at this point.
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 15, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

classes = 2
# TODO: Change this according to the number of classes in the problem.
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = 0
        # TODO: Fill in the matrix Z which will represent the probability distribution of every point.

# Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
