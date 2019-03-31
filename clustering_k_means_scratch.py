import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):  # tol: how much centroid moves in % change before terminating
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for iter in range(self.max_iter):
            self.classifications = {}  # {centroid: [samples in cluster]}
            for i in range(self.k):
                self.classifications[i] = []

            for sample in data:  # data is X (features list)
                distances = [np.linalg.norm(sample-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(sample)

            prev_centroids = dict(self.centroids)  # we want to compare prev_centroids with self.centroids, if we set prev_centroids = self.centroids, they would always be equal, and changes to one would change the other

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for centroid in self.centroids:
                prev_centroid = prev_centroids[centroid]
                curr_centroid = self.centroids[centroid]
                # print(type(curr_centroid))
                shift = abs(np.sum((curr_centroid-prev_centroid)/prev_centroid*100.0))
                if shift > self.tol:  # subtracting ndarrays returns an ndarray with element-wise-subracted elements
                    print('Shift: {}%'.format(shift))
                    optimized = False

            if optimized:
                print('Converged in {} iterations'.format(iter+1))
                break

    def predict(self, data):  # not predicting individual samples (feature sets), but rather the whole data set (clustering)
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = ['g', 'r', 'c', 'b', 'k', 'm', 'y']

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for sample in clf.classifications[classification]:
        plt.scatter(sample[0], sample[1], marker='x', color=color, s=150, linewidths=5)

unknowns = np.array([[1, 3],
                    [8, 9],
                    [0, 3],
                    [5, 4],
                    [6, 4]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=150, linewidths=5)

plt.show()
