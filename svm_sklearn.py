import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)  # missing attribute values denoted by '?'
df.drop(['id'], 1, inplace=True)  # id has no correlation with a tumor being benign or malignant

X = np.array(df.drop(['class'], 1))  # features
y = np.array(df['class'])  # labels

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()  # can use parameter n_jobs=-1 to use as many threads as possible
clf.fit(X_train, y_train)
with open('svm.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('svm.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print('Accuracy: {}'.format(accuracy))

# example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])  # 2 samples
# example_measures = example_measures.reshape(len(example_measures), -1)
# prediction = clf.predict(example_measures)
# print(prediction)
