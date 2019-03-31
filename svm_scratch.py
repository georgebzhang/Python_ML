import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}  # red for samples in class 1, blue for samples in class -1
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)  # 1x1 grid, 1st (and only) spot

    def fit(self, data):
        self.data = data
        opt_dict = {}  # optimization dict {||w||: [w, b]}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]
        all_data = []
        for yi in self.data:  # yi is class (1 or -1)
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None  # releasing memory

        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value*0.01,
                      self.max_feature_value*0.001]

        b_range_multiple = 5  # extremely expensive

        b_multiple = 5

        latest_optimum = self.max_feature_value*10  # cutting major corners

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False  # convex problem allows us to do this
            while not optimized:
                pass

    def predict(self, data):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

# -1 and 1 are classes
data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])}
