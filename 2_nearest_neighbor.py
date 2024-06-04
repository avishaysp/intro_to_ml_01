import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
sections = [
    'c',
    'd',
]

# split training and test

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


# sections (a) + (b) - Implementation

def knn(train, labels, image, k):
    distances = calc_distances(train, image)
    nn_labels = nearest_neighbors(distances, k, labels)

    return most_common_label(nn_labels)


def calc_distances(train, image):
    return np.linalg.norm(np.array(train) - np.array(image), axis=1)


def nearest_neighbors(distances, k, labels):
    nn_indexes = distances.argsort()[:k]
    nn_labels = [labels[index] for index in nn_indexes]
    return nn_labels


def most_common_label(nn_labels):
    label_counters = dict()
    for label in nn_labels:
        if label not in label_counters:
            label_counters[label] = 1
        else:
            label_counters[label] += 1
    most_common_label = max(label_counters, key=label_counters.get)
    return most_common_label


# Use the first 1000 training images


def calc_accuracy(params):
    k, n = params
    trimmed_train_data = train[:n]
    trimmed_train_labels = train_labels[:n]
    correct_predictions = 0
    for i in range(len(test)):
        prediction = knn(trimmed_train_data, trimmed_train_labels, test[i], k=k)
        if prediction == test_labels[i]:
            correct_predictions += 1
    # Calculate the percentage of correct classifications
    return (correct_predictions / len(test)) * 100


# sections (c)

if 'c' in sections:
    print(f"Accuracy for using only 1000 samples in knn is: {calc_accuracy((10, 1000))}%")
    params = [(k, 1000) for k in range(1, 101)]
    ks = [k for k in range(1, 101)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        accuracies = list(executor.map(calc_accuracy, params))

    plt.plot(ks, accuracies, label='Accuracy')
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy as a function of number of neighbors')
    plt.legend()
    plt.show()


# section (d)
if 'd' in sections:
    params = [(3, n) for n in range(100, 5001, 100)]
    ns = [n for n in range(100, 5001, 100)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        accuracies = list(executor.map(calc_accuracy, params))

    plt.plot(ns, accuracies, label='Accuracy')
    plt.xlabel('Train Data Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy as a Function of Train Data Size')
    plt.legend()
    plt.show()


