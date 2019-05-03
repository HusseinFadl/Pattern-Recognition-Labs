import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def generate_C1_points(number_of_points):
    C = []
    for i in range(number_of_points):
        rand = np.random.randint(0, 5)
        if rand == 0:
            C.append(np.random.normal(0, 2))
        elif rand == 1:
            C.append(np.random.normal(2, 3))
        elif rand == 2:
            C.append(np.random.normal(-3, 1))
        elif rand == 3:
            C.append(np.random.standard_exponential())
        else:
            C.append(np.random.binomial(30, 0.2))

    return np.array(C)


def generate_C2_points(number_of_points):
    C = []
    for i in range(number_of_points):
        rand = np.random.randint(0, 4)
        if rand == 0:
            C.append(np.random.normal(10, 2))
        elif rand == 1:
            C.append(np.random.normal(12, 1))
        elif rand == 2:
            C.append(np.random.normal(9, 5))
        # elif (rand ==3):
        #     C.append( np.random.standard_exponential())
        else:
            C.append(np.random.binomial(100, 0.2))

    return np.array(C)


def generate_data():
    number_of_c1_points = 1000
    np.random.seed(10)
    class1_points = generate_C1_points(number_of_c1_points)
    c1 = np.vstack((1 * np.ones(class1_points.shape, dtype='int'), class1_points)).T
    x_test = generate_C1_points(150).T
    c1_test = np.vstack((1 * np.ones(x_test.shape), x_test)).T

    number_of_c2_points = 1500
    x = generate_C2_points(number_of_c2_points)
    c2 = np.vstack((2 * np.ones(x.shape, dtype='int'), x)).T
    x_test = generate_C2_points(150).T
    c2_test = np.vstack((2 * np.ones(x_test.shape), x_test)).T

    c = np.vstack((c1, c2))
    c_test = np.vstack((c1_test, c2_test))
    np.random.shuffle(c)
    np.random.shuffle(c_test)
    with open('data.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c)
    writeFile.close()
    with open('test_data.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c_test[:, 1:])
    writeFile.close()
    with open('test_data_true.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(c_test[:, 0:1])
    writeFile.close()


def read_data():
    with open('data.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
        return data


def read_test_data():
    with open('test_data.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        data = list(reader)
    data_true = []
    with open('test_data_true.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            data_true.append(float(row[0]))
    return data, data_true


# variance : variance of the distribution
# N : number of Dimentions (features)
# M : number of data points of the class
def calculate_h(variance, N, M):
    return np.sqrt(variance) * (4 / ((2 * N + 1) * M)) ** (1 / (N + 4))


# point : point of the class at which we are construction the bump destribution
# x : the point at which we want to know the value of the bump distribution
# h : standard deviation of the bump distribution
# Note this fucntion can be used vectorized
def bump_function(point, x, h):
    return np.exp((np.power(x - point, 2) * -1) / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


generate_data()

data = read_data()
test_data, test_data_true = read_test_data()

X = []
Y = []
c1 = []
c2 = []

for row in data:
    classification = int(float(row[0]))
    x_value = float(row[1])
    Y.append(classification)
    X.append(x_value)
    if classification == 1:
        c1.append(x_value)
    elif classification == 2:
        c2.append(x_value)

XTest = []
for row in test_data:
    XTest.append(float(row[0]))

XTest = np.array(XTest)
c1 = np.asarray(c1)
c2 = np.asarray(c2)
X = np.asarray(X)
Y = np.asarray(Y)

# plot the probability distributions of each class
sns.kdeplot(c1, shade=True)
sns.kdeplot(c2, shade=True)
plt.show()

# calculate h for each class
h1 = calculate_h(np.var(c1), 1, c1.shape[0])
h2 = calculate_h(np.var(c2), 1, c2.shape[0])

# construct x-axis that will be used in density estimation

# get limits of the feature x
min_x = np.min(np.concatenate((c1, c2)))
max_x = np.max(np.concatenate((c1, c2)))

number_of_axis_points = 10000
x = np.linspace(min_x, max_x, number_of_axis_points)

# the probability of each value on the axis
px1 = np.zeros(number_of_axis_points)  # class 1
px2 = np.zeros(number_of_axis_points)  # class 2

for point in c1:
    px1 += bump_function(point, x, h1)

for point in c2:
    px2 += bump_function(point, x, h2)

pc1 = c1.shape[0] / (c1.shape[0] + c2.shape[0])
pc2 = c2.shape[0] / (c1.shape[0] + c2.shape[0])

px1 /= (c1.shape[0] * h1)
px2 /= (c2.shape[0] * h2)

pc1GivenX = px1 * pc1
pc2GivenX = px2 * pc2

# classify points
correct_count = 0
for idx, point in enumerate(XTest):
    index = find_nearest(x, point)
    classification = np.argmax(np.asarray([px1[index], px2[index]])) + 1
    if (classification == test_data_true[int(idx)]):
        correct_count += 1
    print("for point:", point, "the classification:", classification)

print("acc: ", correct_count / XTest.shape[0] * 100, "%")

# plot the estimated densities
plt.plot(x, px1, c='b')
plt.plot(x, px2, c='r')
plt.show()
