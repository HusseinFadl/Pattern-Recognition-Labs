import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import csv
import math

mean1 = np.array([4, -1.3])
cov1 = np.array([[8, 0], [0, 8]])

mean2 = np.array([-5, 3])
cov2 = np.array([[8, 0], [0, 8]])

#Parameters to set
def generate_data():
    x, y = np.random.multivariate_normal(mean1, cov1, 250).T
    c1 = np.vstack((np.ones(x.shape),x,y)).T

    x, y = np.random.multivariate_normal(mean2, cov2, 500).T
    c2 = np.vstack((2*np.ones(x.shape),x,y)).T

    c = np.vstack((c1,c2))
    np.random.shuffle(c)
    with open('data2.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c)
    writeFile.close()

class_counts = [0, 0]
def read_data():
    with open('data2.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        return data

generate_data()
data = read_data()
X = []
Y = []
c1 = []
c2 = []
total_counts = len(data)
for row in data:
    classification = int(float(row[0]))
    x_value = float(row[1])
    y_value = float(row[2])
    Y.append(classification)
    X.append((x_value, y_value))
    if classification == 1:
        c1.append((x_value,y_value))
        class_counts[0]+=1
    else:
        c2.append((x_value, y_value))
        class_counts[1]+=1

c1 = np.asarray(c1)
c2 = np.asarray(c2)
X = np.asarray(X)
Y = np.asarray(Y)

colors = [ 'r' , 'g' , 'b' , 'c' , 'y']
classesCovs = []
classesMeans = []
for classIndex in range(2):
    mask = Y==(classIndex+1)
    XMasked = X[mask]
    plt.scatter(XMasked[:,0] , XMasked[:,1]  ,c = colors[classIndex])
#plt.show()

def multivariate_normal_gaussian(X: np.array, mu: np.array, sigma:np.array ):
    constant = math.pow(1/math.sqrt(2*math.pi),X.shape[0])
    det = np.linalg.det(sigma)
    constant = constant/math.sqrt(det)
    power = -0.5 * (X-mu)@ np.linalg.inv(sigma) @ (X-mu).T
    return np.exp(power)/constant

def draw_decision_boundary(mu1:np.array, mu2:np.array, sigma:np.array, c1, c2):
    w = 2 * np.linalg.inv(sigma) @(mu2-mu1)
    w0 = -mu2.T @ np.linalg.inv(sigma) @ mu2 + mu1.T @ np.linalg.inv(sigma) @ mu1 - 2*np.log10(c1/c2)
    return w, w0

#Create grid and multivariate normal
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(x, y)

Z = np.zeros(X.shape)
Z3,_ = np.meshgrid(np.linspace(0, 50, 300), y)
for i in range (Z.shape[0]):
    for j in range (Z.shape[1]):
        Z[i, j] = multivariate_normal_gaussian(np.array((X[i,j], Y[i,j])), mean1, cov1) +\
                  multivariate_normal_gaussian(np.array((X[i,j], Y[i,j])), mean2, cov2)

w, w0 = draw_decision_boundary(mean1, mean2, cov1, 1/3, 2/3)
X3 = -(w[1]*Y + w0)/w[0]

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, zorder=0.3)
ax.plot_surface(X3, Y, Z3, cmap='plasma', linewidth=0, antialiased=True, zorder=0.8, alpha=0.9)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()