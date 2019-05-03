import numpy as np
import matplotlib.pyplot as plt


def read_data():
    pass


def read_test_data():
    pass


#Note: Pass the suitable parameters
def calculate_h_optimal():
    return 0

def bump_function(x, p, h):
    pass

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

data = read_data()
test_data, test_data_true = read_test_data()

#TODO: Calculate the optimal h for each class

#TODO: Construct x-axis that will be used in density estimation

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
