
import os
# Fix for "Could not load the Qt platform plugin xcb"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
# Force non-interactive backend
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import time
import math
import cv2

from sensor import *
from graph_generator import *
from node import *
from parameter import *
from data_loader import DataLoader

if not train_mode:
    from test_parameter import *

class Env():
    def __init__(self, map_index, k_size=20, plot=False, test=False):
        self.test = test
        self.k_size = k_size
        self.plot = plot
        
        # DataLoader
        self.data_loader = DataLoader()
        if not self.data_loader.map_list:
            raise RuntimeError("No maps found in output/manual_maps!")
            
        self.ground_truth_data = self.data_loader.get_random_map()
        self.ground_truth = self.ground_truth_data['geometric']
        self.semantic_map = self.ground_truth_data['semantic']
        self.map_size = self.ground_truth.shape 
        
        # Robot State (Single Agent)
        self.robot_position = self._find_valid_start_position()
        self.robot_belief = np.full(self.map_size, 127, dtype=np.uint8)
        
        # Graph Generator
        # sensor_range: 3.5m / 0.05 resolution = 70 pixels
        self.sensor_range = 70 
        self.graph_generator = Graph_generator(self.map_size, self.k_size, self.sensor_range, plot)
        
        self.old_robot_belief = self.robot_belief.copy()
        self.explored_rate = 0
        self.done = False
        
        # Initial Sense
        self.update_robot_belief(self.robot_position, self.sensor_range, self.robot_belief, self.ground_truth)
        
        # History
        self.robot_path = [self.robot_position]
        self.travel_dist = 0
        self.steps = 0
        self.found_semantics = set()

        # Initial Graph
        self.update_graph()



    def update_graph(self):
        self.frontiers = self.find_frontier()
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.node_k_flags = \
            self.graph_generator.generate_graph([self.robot_position], self.robot_belief, self.frontiers, k_traj=[self.robot_path])

    def _find_valid_start_position(self):
        for _ in range(1000):
            r = np.random.randint(0, self.map_size[0])
            c = np.random.randint(0, self.map_size[1])
            if self.ground_truth[r, c] == 255: 
                return np.array([c, r], dtype=float) # Return (x, y)
        return np.array([self.map_size[1]//2, self.map_size[0]//2], dtype=float)

    def step(self, next_position):
        # Move
        self.robot_position = next_position
        dist = np.linalg.norm(self.robot_position - self.robot_path[-1])
        self.travel_dist += dist
        self.robot_path.append(next_position)
        self.steps += 1
        
        # Sense
        self.update_robot_belief(next_position, self.sensor_range, self.robot_belief, self.ground_truth)
        
        # Semantic Update
        new_semantics = self.update_semantic_status(next_position)
        
        return new_semantics, dist

    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        # robot_position is (x, y), sensor_work expects (x, y)
        self.robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        
    def update_semantic_status(self, robot_position):
        x, y = int(robot_position[0]), int(robot_position[1]) 
        r = self.sensor_range
        y_min = max(0, y - r); y_max = min(self.map_size[0], y + r)
        x_min = max(0, x - r); x_max = min(self.map_size[1], x + r)
        
        local_belief = self.robot_belief[y_min:y_max, x_min:x_max]
        local_sem = self.semantic_map[y_min:y_max, x_min:x_max]
        
        H, W = local_belief.shape
        cy, cx = y - y_min, x - x_min
        Y, X = np.ogrid[:H, :W]
        dist_sq = (X - cx)**2 + (Y - cy)**2
        mask_circle = dist_sq <= r**2
        
        mask_visible = (local_belief == 255) & mask_circle
        visible_ids = np.unique(local_sem[mask_visible])
        
        new_count = 0
        for uid in visible_ids:
            if uid > 0 and uid not in self.found_semantics:
                self.found_semantics.add(uid)
                new_count += 1
        return new_count

    def calculate_reward(self, move_dist):
        old_free_count = np.sum(self.old_robot_belief == 255)
        new_free_count = np.sum(self.robot_belief == 255)
        reward = (new_free_count - old_free_count) * 1.0 
        reward -= move_dist * 0.05 
        
        self.old_robot_belief = self.robot_belief.copy()
        return reward

    def check_done(self):
        if len(self.frontiers) == 0:
            self.done = True
        return self.done
        
    def find_frontier(self):
        return get_frontier_in_map(self.robot_belief)

    def plot_env(self, global_step, path, step_idx):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
    def plot_env(self, global_step, path, step_idx):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        # Visualize Robot Belief
        map_vis = np.zeros_like(self.robot_belief, dtype=float)
        map_vis[self.robot_belief == 255] = 1.0
        map_vis[self.robot_belief == 1] = 0.0
        map_vis[self.robot_belief == 127] = 0.5
        
        ax.imshow(map_vis, cmap='gray', vmin=0, vmax=1)
        
        # Plot Path (x, y)
        path_x = [p[0] for p in self.robot_path]
        path_y = [p[1] for p in self.robot_path]
        ax.plot(path_x, path_y, 'b-', linewidth=1, label='Path')
        ax.plot(path_x[-1], path_y[-1], 'bo', markersize=5, label='Robot') # Current pos
        
        # Plot Frontiers (x, y)
        if len(self.frontiers) > 0:
            ax.scatter(self.frontiers[:, 0], self.frontiers[:, 1], c='r', s=2, label='Frontiers')
        
        # Plot Nodes (x, y)
        if self.plot and self.node_coords is not None:
             ax.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c='g', s=2, alpha=0.5, label='Graph Nodes')
             
             # Visualize Reachable Nodes (Neighbors)
             current_idx = self.find_index_from_coords(self.robot_position)
             if current_idx is not None and current_idx < len(self.node_coords):
                 # Get neighbors from graph
                 # self.graph is a dict: node_idx -> [neighbor_indices]
                 # Keys/Values might be strings or ints, check graph implementation. 
                 # Assuming Graph_generator.graph.edges is dict of int->list
                 str_idx = str(current_idx)
                 if str_idx in self.graph:
                     neighbors = self.graph[str_idx]
                     for n_idx in neighbors:
                         n_idx = int(n_idx)
                         if n_idx < len(self.node_coords):
                             n_pos = self.node_coords[n_idx]
                             # Draw line
                             ax.plot([self.robot_position[0], n_pos[0]], 
                                     [self.robot_position[1], n_pos[1]], 
                                     c='cyan', linewidth=0.5, alpha=0.8)
                             ax.scatter(n_pos[0], n_pos[1], c='cyan', s=10, marker='x')

        ax.legend(loc='upper right')
        ax.set_title(f'Episode {global_step} Step {step_idx} | Semantic Gain: {len(self.found_semantics)}')
        
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(f'{path}/step_{step_idx:04d}.png')
        plt.close(fig)

    def save_ground_truth(self, path):
         fig = plt.figure(figsize=(10, 10))
         ax = fig.add_subplot(111)
         
         # Ground Truth: 255=Free, 1=Obstacle
         # Note: Ground truth is NOT dilated, it's the actual map
         map_vis = np.zeros_like(self.ground_truth, dtype=float)
         map_vis[self.ground_truth == 255] = 1.0
         map_vis[self.ground_truth == 1] = 0.0
         # 127 might be in ground truth if unmapped, but usually GT is 0/1. 
         # Checking data_loader: normalized_map[geo==255]=0, [geo==0]=1, else 127.
         map_vis[self.ground_truth == 127] = 0.5
         
         ax.imshow(map_vis, cmap='gray', vmin=0, vmax=1)
         ax.set_title("Ground Truth Global Map")
         
         if not os.path.exists(path):
            os.makedirs(path)
         fig.savefig(f'{path}/ground_truth.png')
         plt.close(fig)

    # Added proxy method to fix AttributeError
    def find_index_from_coords(self, coords):
        return self.graph_generator.find_index_from_coords(self.node_coords, coords)


def get_frontier_in_map(map_data):
    free_mask = (map_data == 255).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(free_mask, kernel)
    
    unknown_mask = (map_data == 127)
    frontier_mask = (dilated == 1) & unknown_mask
    
    y, x = np.where(frontier_mask)
    if len(y) == 0:
        return np.zeros((0, 2))
    return np.stack([x, y], axis=1) # Return (x, y)
