import copy
import os

import imageio
import numpy as np
import torch
from env import Env
from parameter import *
import time


class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, num_robots=N_ROBOTS)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dists = [0 for _ in range(N_ROBOTS)]
        self.robot_positions = [self.env.start_position for _ in range(N_ROBOTS)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(17):
            self.episode_buffer.append([])

    def get_observations(self, i_robot):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)
        node_k_flags = copy.deepcopy(self.env.node_k_flags)

        # normalize observations
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1 + int(USE_K_FLAGS) * N_ROBOTS + int(USE_C)))
        if USE_K_FLAGS:
            # print('node_coords:', node_coords.shape, 'node_utility_inputs:', node_utility_inputs.shape, 'guidepost:', guidepost.shape, 'node_k_flags:', node_k_flags.shape)
            node_inputs = np.concatenate((node_coords, node_utility_inputs), axis=1)
            if USE_GUIDEPOST:
                node_inputs = np.concatenate((node_inputs, guidepost), axis=1)
        else:
            node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size, 4)

        # padding the number of node to a given node padding size
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # get the node index of the current robot position
        # current_node_index = self.env.find_index_from_coords(self.robot_position)
        # current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        current_node_index = self.env.find_index_from_coords(self.robot_positions[i_robot])
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1, 1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        attention_matrix = self.calculate_attention_matrix(node_utility) if USE_K_FLAGS else adjacent_matrix
        
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        utility_mask = torch.from_numpy(attention_matrix).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)
        utility_mask = padding(utility_mask)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(-1)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
        
        # calculate a mask for the padded edges (denoted by -1)
        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == -1, one, edge_padding_mask)
        edge_inputs = torch.where(edge_inputs == -1, 0, edge_inputs)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask
        return observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask, utility_mask, self.greedy)
        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
        next_node_index = edge_inputs[0, 0, action_index]
        next_position = self.env.node_coords[next_node_index]
        # print('action_indexes:', action_indexes)
        # print('next_position:', next_positions)
        return next_position, action_index

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[4] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[5] += copy.deepcopy(edge_mask).bool()

        self.episode_buffer[15] += copy.deepcopy(utility_mask).bool()


    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[8] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask = observations
        self.episode_buffer[9] += copy.deepcopy(node_inputs)
        self.episode_buffer[10] += copy.deepcopy(edge_inputs)
        self.episode_buffer[11] += copy.deepcopy(current_index)
        self.episode_buffer[12] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[13] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[14] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[16] += copy.deepcopy(utility_mask).bool()


    def run_episode(self, curr_episode):
        done = False
        
        for i in range(128):
            # start = time.time()
            # print("epi",i)
            for i_robot in range(N_ROBOTS):
                observations = self.get_observations(i_robot)
                self.save_observations(observations)
                
                next_position, action_indexes = self.select_node(observations)

                self.save_action(action_indexes)
                reward, done, self.robot_positions[i_robot], self.travel_dists[i_robot] = self.env.step(self.robot_positions[i_robot], next_position, self.travel_dists[i_robot], i_robot)
                self.save_reward_done(reward, done)
    
                observations = self.get_observations(i_robot)

                self.save_next_observations(observations)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, sum(self.travel_dists))
                self.env.plot_iou(self.global_step, gifs_path, i, sum(self.travel_dists))
                self.env.plot_scatter_env(self.global_step, gifs_path, i, sum(self.travel_dists))

            if done:
                break
            # print("epi time", time.time()-start)

        # save metrics
        self.perf_metrics['travel_dist'] = sum(self.travel_dists)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['total_steps'] = i
        self.perf_metrics['explored_IOU'] = self.env.final_iou

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)
        

    def work(self, currEpisode):
        self.run_episode(currEpisode)

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            cnt = 0
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
                    cnt += 1
            assert cnt <= K_SIZE
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



