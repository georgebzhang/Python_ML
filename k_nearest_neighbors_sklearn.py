import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle

accuracies = []

num_runs = 25
for i in range(num_runs):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)  # missing attribute values denoted by '?'
    df.drop(['id'], 1, inplace=True)  # id has no correlation with a tumor being benign or malignant

    X = np.array(df.drop(['class'], 1))  # features
    y = np.array(df['class'])  # labels

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()  # can use parameter n_jobs=-1 to use as many threads as possible
    clf.fit(X_train, y_train)
    with open('knearestneighbors.pickle', 'wb') as f:
        pickle.dump(clf, f)

    pickle_in = open('knearestneighbors.pickle', 'rb')
    clf = pickle.load(pickle_in)

    accuracy = clf.score(X_test, y_test)
    # print(accuracy)

    # example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])  # 2 samples
    # example_measures = example_measures.reshape(len(example_measures), -1)
    # prediction = clf.predict(example_measures)
    # print(prediction)

    accuracies.append(accuracy)

avg_accuracy = sum(accuracies)/len(accuracies)
print('Average accuracy: {}'.format(avg_accuracy))