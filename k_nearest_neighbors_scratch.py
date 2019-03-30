import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups!')
    distances = []
    for group in data:  # group, i.e. class
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature)-np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


# 'k': features that correspond to class 'k'
# 'r': features that correspond to class 'r'
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# for cls in dataset:  # cls indicates class
#     for feat in dataset[cls]:  # feat indicates feature
#         plt.scatter(feat[0], feat[1], s=100, color=cls)

[[plt.scatter(feat[0], feat[1], s=100, color=cls) for feat in dataset[cls]] for cls in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()