import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(x, y, z,  title='', xlabel='', ylabel='', zlabel='',color_style_str='', label_str='', figure=None, axis=None):
    if figure is None:
        fig = plt.figure()
    else:
        fig = figure
    ax = axis

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.scatter(x, y, z, color = color_style_str, label=label_str)
    handles, labels = ax.get_legend_handles_labels()

    unique = list(set(labels))
    handles = [handles[labels.index(u)] for u in unique]
    labels = [labels[labels.index(u)] for u in unique]

    ax.legend(handles, labels)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(gray, (3, 3))    # blur the image to remove the noise
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    return thresh

def findContourArea(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[1])
    return area, contours

def findBoundingRectangleArea(img, contours):
    x, y, w, h = cv2.boundingRect(contours[1])
    bounding_rectangle = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    area = w*h
    return area, bounding_rectangle

def findBoundingCircleArea(img, contours):
    (x, y), radius = cv2.minEnclosingCircle(contours[1])
    center = (int(x), int(y))
    radius = int(radius)
    bounding_circle = cv2.circle(img.copy(), center, radius, (0, 255, 0), 2)
    area = radius * radius * math.pi
    return area, bounding_circle

def findBoundingTriangleArea(img, contours):
    x = cv2.minEnclosingTriangle(contours[1])
    bounding_triangle = cv2.polylines(img.copy(), np.int32([x[1]]), True, (0, 255, 0), 2)
    return x[0], bounding_triangle

def calculateDistance(x1,x2):
    return np.linalg.norm(x2-x1);

def MinimumDistanceClassifier(test_point, all_points):
    c1 = all_points[all_points[:,0] == 1 ]
    c2 = all_points[all_points[:,0] == 2 ]
    c3 = all_points[all_points[:,0] == 3 ]

    mean1 = np.mean(c1[:,1:],axis = 0)
    mean2 = np.mean(c2[:,1:],axis = 0)
    mean3 = np.mean(c3[:,1:],axis = 0)

    distance1 = calculateDistance(mean1, test_point)
    distance2 = calculateDistance(mean2, test_point)
    distance3 = calculateDistance(mean3, test_point)

    classification = np.argmin((distance1, distance2, distance3)) + 1 # the + 1 because it returns a zero based index
    return classification

def NearestNeighbor(test_point, all_points):
    min = math.inf
    min_class = 0
    for point in all_points:
        dist = calculateDistance(test_point, point[1:])
        if (dist < min):
            min = dist
            min_class = point[0]

    return min_class

def KNN(test_point, all_points, k):
    distances = []
    classes = []
    for point in all_points:
        dist = calculateDistance(test_point, point[1:])
        distances.append(dist)
        classes.append(point[0])

    classes = np.array(classes)
    sorted_indices = np.argsort(distances)
    nearest_indices = sorted_indices[0:k]
    nearest_classes = classes[nearest_indices].astype('int')
    return np.argmax(np.bincount(nearest_classes))


def get_class_from_file_name(file_name):
    return file_name.split("test\\")[1].split(".")[0]


def get_class_name(class_number):
    classes = ["", "Rectangle", "Circle", "Triangle"]
    return classes[int(class_number)]


def extract_features(img, class_number=None):
    area, contours = findContourArea(img)
    area1,_ = findBoundingRectangleArea(img, contours)
    area2,_ = findBoundingCircleArea(img, contours)
    area3,_ = findBoundingTriangleArea(img, contours)
    if class_number is None:
        features = [area/area1, area/area2, area/area3]
    else:
        features = [class_number, area / area1, area / area2, area / area3]
    return features

training_data = []
training_data_rec = []
training_data_circle = []
training_data_tri = []

for filename in glob.glob('images/rectangle/*.png'):
    img = cv2.imread(filename)
    img = preprocess(img)
    img_features = extract_features(img, 1)
    training_data.append(img_features)
    training_data_rec.append(img_features)

for filename in glob.glob('images/circle/*.png'):
    img = cv2.imread(filename)
    img = preprocess(img)
    img_features = extract_features(img, 2)
    training_data.append(img_features)
    training_data_circle.append(img_features)

for filename in glob.glob('images/tri/*.png'):
    img = cv2.imread(filename)
    img = preprocess(img)
    img_features = extract_features(img, 3)
    training_data.append(img_features)
    training_data_tri.append(img_features)

training_data = np.asarray(training_data)
training_data_rec = np.asarray(training_data_rec)
training_data_circle = np.asarray(training_data_circle)
training_data_tri = np.asarray(training_data_tri)

fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')

plot(training_data_rec[:, 1], training_data_rec[:, 2], training_data_rec[:, 3], title='Training Data',
     xlabel='Feature Rec.', ylabel='Feature Circle', zlabel='Feature Tri.', color_style_str='r', label_str = "Rectangle",
     figure=fig, axis=ax)

plot(training_data_circle[:, 1], training_data_circle[:, 2], training_data_circle[:, 3], title='Training Data',
     xlabel='Feature Rec.', ylabel='Feature Circle', zlabel='Feature Tri.', color_style_str='b', label_str = "Circle",
     figure=fig, axis=ax)

plot(training_data_tri[:, 1], training_data_tri[:, 2], training_data_tri[:, 3], title='Training Data',
     xlabel='Feature Rec.', ylabel='Feature Circle', zlabel='Feature Tri.', color_style_str='g', label_str = "Triangle",
     figure=fig, axis=ax)

plt.show()

true_values = [3, 1, 1, 3, 3, 1, 1, 2, 3, 2]

min_distance_predictions = []
nn_predictions = []
knn_predictions = []
index = 0
for filename in glob.glob('test/*.png'):
    img_original = cv2.imread(filename)
    img = preprocess(img_original)
    test_point = extract_features(img)
    print("Actual class :", get_class_name(true_values[index]))
    print("---------------------------------------")
    index += 1

    min_dist_prediction = MinimumDistanceClassifier(test_point, training_data)
    nn_prediction = NearestNeighbor(test_point, training_data)
    knn_prediction = KNN(test_point, training_data, 3)

    print("Minimum Distance Classifier Prediction   :", get_class_name(min_dist_prediction))
    print("Nearest Neighbour Prediction             :", get_class_name(nn_prediction))
    print("K-Nearest Neighbours Prediction          :", get_class_name(knn_prediction))
    print("===========================================================================")
    min_distance_predictions.append(min_dist_prediction)
    nn_predictions.append(nn_prediction)
    knn_predictions.append(knn_prediction)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_original)
    cv2.waitKey(0) # this means to go to the next image press any key
    cv2.destroyAllWindows()

print("Minimum Distance Classifier Accuracy: ",
      np.sum((np.array(min_distance_predictions) == true_values).astype('int')) / len(true_values) * 100, "%")
print("Nearest Neighbour Classifier Acccuracy: ",
      np.sum((np.array(nn_predictions) == true_values).astype('int')) / len(true_values) * 100, "%")
print("K-Nearest Neighbour Classifier Accuracy: ",
      np.sum((np.array(knn_predictions) == true_values).astype('int')) / len(true_values) * 100, "%")