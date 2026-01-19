import numpy as np
from parameter import *
if not train_mode:
    from test_parameter import *
from sklearn.cluster import DBSCAN

class Node():
    def __init__(self, coords, frontiers, robot_belief):
        self.coords = coords
        self.dbscan = DBSCAN(eps=10, min_samples=3)
        self.observable_frontiers = []
        self.sensor_range = 70
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility = self.get_node_utility()
        self.clusters = self.get_node_clusters()
        self.k_visit = [0 for i in range(N_ROBOTS)]
        self.k_curr = [0 for i in range(N_ROBOTS)]
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def initialize_observable_frontiers(self, frontiers, robot_belief):
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < self.sensor_range - 5]
        for point in frontiers_in_range:
            collision = self.check_collision(self.coords, point, robot_belief)
            if not collision:
                self.observable_frontiers.append(point)

    def get_node_utility(self):
        utility = len(self.observable_frontiers)
        return utility if utility > 2 else 0

    def get_node_clusters(self):
        if len(self.observable_frontiers) > 0:
            clusters = self.dbscan.fit_predict(np.array(self.observable_frontiers))
            unique_labels = set(clusters)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            return len(unique_labels)
        else:
            return 0

    def update_observable_frontiers(self, observed_frontiers, new_frontiers, robot_belief):
        # remove observed frontiers in the observable frontiers
        if observed_frontiers != []:
            observed_index = []
            for i, point in enumerate(self.observable_frontiers):
                if point[0] + point[1] * 1j in observed_frontiers[:, 0] + observed_frontiers[:, 1] * 1j:
                    observed_index.append(i)
            for index in reversed(observed_index):
                self.observable_frontiers.pop(index)

        # add new frontiers in the observable frontiers
        if new_frontiers != []:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.sensor_range - 10]
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.append(point)
        self.clusters = self.get_node_clusters()
        self.utility = self.get_node_utility()
        if self.utility <= 5:
            self.utility = 0
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        self.observable_frontiers = []
        self.utility = 0
        self.clusters = 0
        self.zero_utility_node = True

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision
