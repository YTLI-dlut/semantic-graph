import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from env import Env
from model import PolicyNet
from test_parameter import *
import json


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, test=True, num_robots=N_ROBOTS)
        self.local_policy_net = policy_net
        self.travel_dist = 0
        self.robot_position = self.env.start_position
        self.perf_metrics = dict()
        self.robot_positions = [self.env.start_position for _ in range(N_ROBOTS)]

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(128):
            next_position, action_index = self.select_node(observations)
            explore_rate = self.env.explored_rate
            reward, done, self.robot_positions, self.travel_dist = self.env.step(self.robot_positions, next_position,
                                                                               self.travel_dist)
            if self.env.explored_rate >= 0.9 and explore_rate < 0.9:
                self.perf_metrics['90'] = self.travel_dist
            observations = self.get_observations()

            # save evaluation data
            if SAVE_TRAJECTORY:
                if not os.path.exists(trajectory_path):
                    os.makedirs(trajectory_path)
                csv_filename = f'results/trajectory/ours_trajectory_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist', 'area']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist, np.sum(self.env.combined_belief == 255)]).reshape(1, -1)
                    writer.writerows(csv_data)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)
                self.env.plot_iou(self.global_step, gifs_path, i, self.travel_dist)
                self.env.plot_scatter_env(self.global_step, gifs_path, i, self.travel_dist)
            if done:
                break

        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['steps'] = i
        self.perf_metrics['EIOU'], _ = self.env.belief_iou(self.env.robot_beliefs)
        # self.perf_metrics['k_dist'] = self.env.k_dist

        # with open('trajectory.json', 'w') as f:
        #     # print(self.env.k_traj)
        #     traj = np.array(self.env.k_traj)
        #     json.dump(traj.tolist(), f)

        # save final path length
        if SAVE_LENGTH:
            if not os.path.exists(length_path):
                os.makedirs(length_path)
            csv_filename = f'results/length/ours_length_result.csv'
            new_file = False if os.path.exists(csv_filename) else True
            field_names = ['dist']
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                if new_file:
                    writer.writerow(field_names)
                csv_data = np.array([self.travel_dist]).reshape(-1,1)
                writer.writerows(csv_data)
            
            # csv_filename = f'results/length/ours_90length_result.csv'
            # new_file = False if os.path.exists(csv_filename) else True
            # field_names = ['dist']
            # with open(csv_filename, 'a') as csvfile:
            #     writer = csv.writer(csvfile)
            #     if new_file:
            #         writer.writerow(field_names)
            #     csv_data = np.array([self.perf_metrics['90']]).reshape(-1,1)
            #     writer.writerows(csv_data)

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)

        # normalize observations
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1 + int(USE_K_FLAGS) * N_ROBOTS + int(USE_C)))
        if USE_K_FLAGS:
            # print('node_coords:', node_coords.shape, 'node_utility_inputs:', node_utility_inputs.shape, 'guidepost:', guidepost.shape, 'node_k_flags:', node_k_flags.shape)
            node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost), axis=1)
        else:
            node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size, 4)

        # calculate a mask for padded node
        node_padding_mask = None

        # get the node index of the current robot position
        # current_node_index = self.env.find_index_from_coords(self.robot_position)
        # current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        current_indexes = []
        for i in range(N_ROBOTS):
            current_node_index = self.env.find_index_from_coords(self.robot_positions[i])
            current_indexes.append(current_node_index)
        current_index = torch.tensor(current_indexes).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,2)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        if USE_K_FLAGS:
            attention_matrix = self.calculate_attention_matrix(node_utility)
            utility_mask = torch.from_numpy(attention_matrix).float().unsqueeze(0).to(self.device)
        else:
            utility_mask = None

        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        all_edges_inputs = []
        all_edge_padding_masks = []
        for i in range(N_ROBOTS):
            edge = edge_inputs[current_indexes[i]]
            while len(edge) < self.k_size:
                edge.append(0)

            edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
            all_edges_inputs.append(edge_input)

            # calculate a mask for the padded edges (denoted by 0)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_input == 0, one, edge_padding_mask)
            all_edge_padding_masks.append(edge_padding_mask)
        edge_inputs = torch.cat(all_edges_inputs, dim=-1)
        edge_padding_mask = torch.cat(all_edge_padding_masks, dim=-1)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask = observations
        with torch.no_grad():
            logp_list, action_indexes = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask, utility_mask, self.greedy)
        next_positions = []
        for i in range(N_ROBOTS):
            next_node_index = edge_inputs[0, 0, action_indexes[i]]
            next_position = self.env.node_coords[next_node_index]
            next_positions.append(next_position)
        # print('action_indexes:', action_indexes)
        # print('next_position:', next_positions)
        return next_positions, action_indexes

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix
    
    def calculate_attention_matrix(self, node_utility):
        size = len(node_utility)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for k in range(N_ROBOTS):
                if node_utility[i, k + 1] == 0:
                    for j in range(size):
                        # node with utility > 0
                        if node_utility[j, 0] > 0:
                            bias_matrix[i][j] = 0
                            bias_matrix[j][i] = 0
                        # node with a robot
                        for l in range(N_ROBOTS):
                            if node_utility[j, l + 1] == 0:
                                bias_matrix[i][j] = 0
                                bias_matrix[j][i] = 0
                                break
                    break
        return bias_matrix

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}_iou.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.iou_frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}_scatter.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.scatter_frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)
        for filename in self.env.iou_frame_files[:-1]:
            os.remove(filename)
        for filename in self.env.scatter_frame_files[:-1]:
            os.remove(filename)


    def work(self, curr_episode):
        self.run_episode(curr_episode)
