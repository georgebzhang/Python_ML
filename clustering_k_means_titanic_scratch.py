import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import xlrd
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


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0

            for element in unique_elements:
                if element not in text_digit_vals:
                    text_digit_vals[element] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))
            # df[column] = [convert_to_int(val) for val in df[column]]

    return df


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)

df.drop(['pclass', 'parch', 'fare'], 1, inplace=True)  # testing effect on accuracy by removing features

X = np.array(df.drop(['survived'], 1).astype(float))  # features list
y = np.array(df['survived'])  # labels
X = preprocessing.scale(X)  # increased accuracy by 20%

colors = ['g', 'r', 'c', 'b', 'k', 'm', 'y']

clf = K_Means()
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = clf.predict(predict)
    if prediction == y[i]:
        correct += 1

accuracy = correct/len(X)
print('Accuracy: {}'.format(accuracy))
