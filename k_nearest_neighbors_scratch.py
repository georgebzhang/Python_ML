import numpy as np
# import matplotlib.pyplot as plt
import warnings
# from matplotlib import style
from collections import Counter
# style.use('fivethirtyeight')
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups!')
    distances = []
    for group in data:  # group, i.e. class
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([euclidean_distance, group])  # distances = [[dist, group], ...]
    votes = [i[1] for i in sorted(distances)[:k]]  # i[1] = group
    most_common = Counter(votes).most_common(1)[0]  # most_common(...) returns a list
    vote_result = most_common[0]
    confidence = most_common[1]/k
    return vote_result, confidence


# 'k': features that correspond to class 'k'
# 'r': features that correspond to class 'r'
# dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
# new_features = [5, 7]

# result = k_nearest_neighbors(dataset, new_features, k=3)

# for cls in dataset:  # cls indicates class
#     for feat in dataset[cls]:  # feat indicates feature
#         plt.scatter(feat[0], feat[1], s=100, color=cls)

# [[plt.scatter(feat[0], feat[1], s=100, color=cls) for feat in dataset[cls]] for cls in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, color=result)
# plt.show()

accuracies = []

num_runs = 25
for i in range(num_runs):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()  # ensures that everything in dataframe is float
    random.shuffle(full_data)  # shuffle data

    # print(full_data[:10])

    test_size = 0.2
    train_set = {2: [], 4: []}  # similar to defaultdict(list)
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size*len(full_data))]  # all data excluding last test_size*100%
    test_data = full_data[-int(test_size*len(full_data)):]  # last test_size*100% of data

    for i in train_data:
        train_set[i[-1]].append(i[:-1])  # i[-1] is label (2 for benign, 4 for malignant), i[:-1] is features (i.e. all data excluding the label)

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:  # group is label (2 for benign, 4 for malignant)
        for data in test_set[group]:  # data is features
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)  # increasing k may decrease accuracy, data is unbalanced (65.5% benign, 34.5% malignant)
            if vote == group:
                correct += 1
            # else:
                # print('Confidence of misclassified sample: {}'.format(confidence))
            total += 1

    accuracy = correct/total
    # print('Accuracy: {}'.format(accuracy))
    accuracies.append(accuracy)

avg_accuracy = sum(accuracies)/len(accuracies)
print('Average accuracy: {}'.format(avg_accuracy))