from collections import Counter
import numpy as np

from read_dataset import read_dataset


def distance(x1: np.ndarray, x2: np.ndarray):
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        return np.sum((x1 - x2) ** 2)
    else:
        exit()


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class KNN:
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.vectorize(self.predict_element)(X)

    def predict_element(self, x):
        assert isinstance(x, np.ndarray)
        assert x.shape == (28, 28)

        # Compute distances between x and all examples in the training set
        a = np.reshape(self.X_train, newshape=(-1,))

        distances = [distance(x, x_train) for x_train in a]

        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        b = np.reshape(self.y_train, newshape=(-1,))
        k_neighbor_labels = [b[i] for i in k_idx]

        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)

        return most_common[0][0]


if __name__ == "__main__":
    dataset = read_dataset()

    classifier = KNN(k=10)
    classifier.fit(dataset.train_data, dataset.train_labels)

    predictions = classifier.predict(dataset.test_data)
    print("KNN classification accuracy", accuracy(dataset.test_labels, predictions))
