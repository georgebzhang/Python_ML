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

        # yi(xi.w+b) = 1 for support vectors
        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value*0.01,
                      self.max_feature_value*0.001]

        b_range_multiple = 5  # extremely expensive, since we are not giving b the same optimization treatment (big steps to small steps) as we are giving w

        b_multiple = 5  # don't need to take as small of steps with b as we do w

        latest_optimum = self.max_feature_value*10  # cutting major corners

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False  # convex problem allows us to do this
            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in SVM fundamentally, SMO attempts to address this
                        # constraint function: yi(xi.w+b) >= 1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False
                                    # break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step  # substracts scalar step from each element of vector w

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]  # smallest ||w||
            self.w = opt_choice[0]  # w = [wx, wy]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2  # wx + step*2

        # helpful debugger: shows yi for training x
        for yi in self.data:
            for xi in self.data[yi]:
                print('{}: {}'.format(xi, yi*(np.dot(self.w, xi)+self.b)))

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[yi]) for x in self.data[yi]] for yi in self.data]

        # hyperplane = x.w+b
        def hyperplane(x, w, b, v):  # purely for us to visualize, SVM does not need
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)  # 10% extra space around edge points
        # hyperplane minimum and maximum x
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive support vector hyperplane w.x+b = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')  # 'k' for black

        # negative support vector hyperplane w.x+b = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # decision boundary w.x+b = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')  # 'y--' for yellow dashed

        plt.show()

# -1 and 1 are classes
data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])}

predict = [[0, 10],
           [1, 3],
           [3, 4],
           [3, 5],
           [5, 5],
           [5, 6],
           [6, -5],
           [5, 8]]

clf = Support_Vector_Machine()
clf.fit(data_dict)

for p in predict:
    clf.predict(p)

clf.visualize()
