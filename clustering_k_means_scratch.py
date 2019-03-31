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

        for i in range(self.max_iter):
            self.classifications = {}  # {centroid: [samples in cluster]}
            for j in range(self.k):
                self.classifications[j] = []

            for sample in data:  # data is X (features list)
                distances = [np.linalg.norm(sample-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(sample)

            prev_centroids = dict(self.centroids)  # we want to compare prev_centroids with self.centroids, if we set prev_centroids = self.centroids, they would always be equal, and changes to one would change the other

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)


    def predict(self, data):  # not predicting individual samples (feature sets), but rather the whole data set (clustering)
        pass


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()

colors = ['g.', 'r.', 'c.', 'b.', 'k.', 'm.', 'y.']