import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import csv
import math

class_means = [[-5, -6], [-0.5, 10], [6, 0]]
class_covariances = [[[6, 4], [4, 6]], [[2, 0], [0, 8]], [[5, 0], [0, 5]]]


def generate_data():
    np.random.seed(10)
    mean = class_means[0]
    cov = class_covariances[0]
    x, y = np.random.multivariate_normal(mean, cov, 500).T
    c1 = np.vstack((1*np.ones(x.shape, dtype='int'),x,y)).T
    x_test, y_test = np.random.multivariate_normal(mean, cov, 150).T + np.random.multivariate_normal([0,0],[[3,0],[0,3]], 150).T
    c1_test = np.vstack((1 * np.ones(x_test.shape), x_test, y_test)).T

    mean = class_means[1]
    cov = class_covariances[1]
    x, y = np.random.multivariate_normal(mean, cov, 400).T
    c2 = np.vstack((2*np.ones(x.shape, dtype='int'),x,y)).T
    x_test, y_test = np.random.multivariate_normal(mean, cov, 150).T + np.random.multivariate_normal([0,0],[[3,0],[0,3]], 150).T
    c2_test = np.vstack((2 * np.ones(x_test.shape), x_test, y_test)).T

    mean = class_means[2]
    cov = class_covariances[2]
    x, y = np.random.multivariate_normal(mean, cov, 250).T
    c3 = np.vstack((3 * np.ones(x.shape, dtype='int'), x, y)).T
    x_test, y_test = np.random.multivariate_normal(mean, cov, 150).T + np.random.multivariate_normal([0,0],[[5,1.5],[1.5,5]], 150).T
    c3_test = np.vstack((3 * np.ones(x_test.shape), x_test, y_test)).T

    c = np.vstack((c1,c2 ,c3))
    c_test = np.vstack((c1_test, c2_test, c3_test))
    np.random.shuffle(c)
    np.random.shuffle(c_test)
    with open('data1.csv', 'w' , newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c)
    writeFile.close()
    with open('test_data.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c_test[:,1:])
    writeFile.close()
    with open('test_data_true.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c_test[:,0:1])
    writeFile.close()

class_counts = [0, 0, 0]
def read_data():
    with open('data1.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        return data

def generate_test_data( numberOfTestPoints ):
    XTest = np.zeros((numberOfTestPoints , 2))
    featureMeans = [0 , 0]
    featureVars = [5, 5]
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
    data_true = []
    with open('test_data_true.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            data_true.append(float(row[0]))
    return data, data_true

generate_data()
data = read_data()
# generate_test_data(10)
test_data, test_data_true = read_test_data()
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
    plt.scatter(XMasked[:,0], XMasked[:,1], c = colors[classIndex])
plt.scatter(XTest[:, 0], XTest[:, 1], c = '#000000')
# plt.show()

def multivariate_normal_gaussian(X: np.array, mu: np.array, sigma:np.array ):
    constant = math.pow(1/math.sqrt(2*math.pi),X.shape[0])
    det = np.linalg.det(sigma)
    constant = constant/math.sqrt(det)
    power = -0.5 *  (X-mu)@ np.linalg.inv(sigma) @ (X-mu).T
    #pep = power / constant
    return np.exp(power)/constant


pClasses = []
estimate_means = []
estimate_covariances = []
for classIndex in range(3):
    pClasses.append(np.sum((Y==(classIndex+1)).astype("int")) / Y.shape[0])
    estimate_means.append(np.mean(X[Y==classIndex+1],axis = 0))
    estimate_covariances.append(np.cov(X[Y==classIndex+1].T))

predicted_vals = []
for testPoint in XTest:
    print("*************-------------*************")
    print("for test point:" , testPoint)
    classProbabilities = np.zeros((3))
    for classIndex in range(3):
        pXGivenC = multivariate_normal_gaussian(testPoint , estimate_means[classIndex] , estimate_covariances[classIndex])
        classProbabilities[classIndex] = pXGivenC * pClasses[classIndex]

    predicted_vals.append(np.argmax(classProbabilities) + 1)
    print("point class is :", np.argmax(classProbabilities) + 1)
    print('class probabilities: ', classProbabilities) # the first class is the left most in the scatter plot
#
# test_data_true = np.asarray(test_data_true)
# predicted_vals = np.asarray(predicted_vals).reshape(len(predicted_vals),1)
# correct = 0
# for i in range(len(test_data_true)):
#     if int(test_data_true[i]) == int(predicted_vals[i]):
#         correct = correct + 1
# accuracy = correct / len(test_data_true)
accuracy = np.sum(np.equal(predicted_vals,test_data_true))/ len(predicted_vals)
print('Accuracy = ' + str(accuracy))
plt.show()

#Create grid and multivariate normal
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 15, 300)
X, Y = np.meshgrid(x, y)

pos = np.empty( X.shape +(2,))
pos[:, : , 0] = X
pos[:, :, 1] = Y


Z = np.zeros(X.shape)
cum = 0
for i in range (Z.shape[0]):
    for j in range (Z.shape[1]):
        Z[i, j] = 0
        for classIndex in range(len(class_means)):
            Z[i , j ] += multivariate_normal_gaussian(np.array((X[i,j],Y[i,j])), class_means[classIndex], class_covariances[classIndex])
            #Z[i, j] *= pClasses[classIndex]
            cum += Z[i,j]

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z,cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()