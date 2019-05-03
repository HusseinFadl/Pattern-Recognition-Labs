import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import csv
import math

class_means = [[-5 , -6 ] ,[-0.5, 10], [6, 0]]
class_covs = [[[3, 1.5], [1.5, 5]] , [[2,4],[0,5]] ,[[3, 0], [3, 2]] ]

def generate_data():

    np.random.seed(10)
    mean = class_means[0]
    cov = class_covs[0]
    x, y = np.random.multivariate_normal(mean, cov, 500).T
    c1 = np.vstack((1*np.ones(x.shape),x,y)).T

    mean = class_means[1]
    cov = class_covs[1]
    x, y = np.random.multivariate_normal(mean, cov, 400).T
    c2 = np.vstack((2*np.ones(x.shape),x,y)).T

    mean = class_means[2]
    cov = class_covs[2]
    x, y = np.random.multivariate_normal(mean, cov, 250).T
    c3 = np.vstack((3 * np.ones(x.shape), x, y)).T

    c = np.vstack((c1,c2 ,c3))
    with open('data1.csv', 'w' , newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c)
    writeFile.close()

class_counts = [0, 0, 0]
def read_data():
    with open('data.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        return data

def generate_test_data( numberOfTestPoints ):
    XTest = np.zeros((numberOfTestPoints , 2))
    featureMeans = [0 , 0]
    featureVars = [ 5, 5]
    for featureIndex in range(2):
        featurePoints = np.random.normal(featureMeans[featureIndex] , featureVars[featureIndex]  , numberOfTestPoints )
        XTest[: , featureIndex] = featurePoints
    print('mean', np.mean(XTest , axis=0))
    with open('test_data.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(XTest)
    writeFile.close()

def read_test_data ():
    with open('test_data.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        return data


generate_data()
data = read_data()
generate_test_data(10)
test_data = read_test_data()
X = []
Y = []
c1 = []
c2 = []
c3 = []
for row in data:
    classification = int(float(row[0]))
    x_value = float(row[1])
    y_value = float(row[2])
    Y.append(classification)
    X.append((x_value, y_value))
    if classification == 1:
        c1.append((x_value,y_value))
    elif classification == 2:
        c2.append((x_value, y_value))
    elif classification == 3:
        c3.append((x_value, y_value))

XTest = []
for row in test_data:
    XTest.append([float(row[0]) , float(row[1])])

XTest = np.array(XTest)
c1 = np.asarray(c1)
c2 = np.asarray(c2)
c3 = np.array(c3)
X = np.asarray(X)
Y = np.asarray(Y)

print("max" , XTest)

colors = [ 'r' , 'g' , 'b' , 'c' , 'y']

for classIndex in range(3):
    mask = Y==(classIndex+1)
    XMasked = X[mask]
    plt.scatter(XMasked[:,0] , XMasked[:,1]  ,c = colors[classIndex])
plt.scatter(XTest[: , 0] , XTest[:,1] , c = '#000000')

def multivariate_normal_gaussian(X: np.array, mu: np.array, sigma:np.array ):
    constant = math.pow(1/math.sqrt(2*math.pi),X.shape[0])
    det = np.linalg.det(sigma)
    constant = constant/math.sqrt(det)
    power = -0.5 *  (X-mu)@ np.linalg.inv(sigma) @ (X-mu).T
    pep = power / constant
    return np.exp(pep)


pClasses = []
for classIndex in range(3):
    pClasses.append( np.sum((Y==(classIndex+1)).astype("int")) / Y.shape[0])
for testPoint in XTest:
    print("*************-------------*************")
    print("for test point:" , testPoint)
    classProbabilities  = np.zeros((3))
    for classIndex in range(3):
        meanVector = class_means[classIndex]
        covMatrix = class_covs[classIndex]
        pXGivenC =multivariate_normal_gaussian(testPoint , meanVector , covMatrix)
        classProbabilities[classIndex] = pXGivenC * pClasses[classIndex]

    print("point class is :", np.argmax( classProbabilities) + 1 )
    print('class probabilities: ' , classProbabilities) # the first class is the left most in the scatter plot

plt.show()

#Create grid and multivariate normal
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 15, 300)
X, Y = np.meshgrid(x, y)

pos = np.empty( X.shape +(2,))
pos[:, : , 0] = X
pos[:, :, 1] = Y


Z = np.zeros(X.shape)
for i in range (Z.shape[0]):
    for j in range (Z.shape[1]):
        Z[i, j] = 0
        for classIndex in range(len(class_means)):
            Z[i , j ] += multivariate_normal_gaussian(pos[i, j, :] , class_means[classIndex] , class_covs[classIndex])
#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()