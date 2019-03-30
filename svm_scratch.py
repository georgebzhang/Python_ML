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
