
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

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(17):
            self.episode_buffer.append([])
            
        self.total_semantic_gain = 0

    def get_observations(self):
        # 1. Get raw graph data
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)

        # 2. Normalize
        map_h, map_w = self.env.map_size
        node_coords = node_coords / max(map_h, map_w)
        node_utility = node_utility / 50.0

        # ... Tensor processing ...
        n_nodes = node_coords.shape[0]
        # In Single Agent, node_utility is (N, 1) or simplified.
        # Parameter.py: INPUT_DIM = 3 + int(USE_C) + int(USE_GUIDEPOST) = 3+1+1 = 5
        # node_coords (2) + utility (1) + guidepost (1) + ? 
        # node_utility needs to be reshaped correctly.
        # MAME originally: utility is [util, dist_r1, dist_r2...]
        # Now we only need [util, dist_r1].
        
        # Re-calc utility vector here since graph_generator might still produce old format or we just fix shape
        # In single agent, we treat node_utility just as [utility_value] usually, 
        # plus maybe distance to self.
        
        # For simplicity, let's assume node_utility currently just holds value
        # We need to manually append dist if needed, or rely on graph_generator doing it.
        # graph_generator logic:
        # if USE_K_FLAGS: append dists...
        # else: append utility.
        # logic in graph_generator: self.node_utility.append(utility) -> scalar.
        
        # So node_utility is (N,) scalar array if !USE_K_FLAGS
        
        node_utility = node_utility.reshape((n_nodes, 1))
        
        node_inputs = np.concatenate((node_coords, node_utility, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        # Padding nodes
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # Current index
        current_node_index = self.env.find_index_from_coords(self.env.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)

        # Edges
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        
        # Utility mask (Attention matrix) - simplified for single agent usually just ajancency
        utility_mask = edge_mask # Or ones

        # Padding edges
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)
        # utility_mask = padding(utility_mask) 

        edge = edge_inputs[current_node_index]
        # Clip to K_SIZE if data has more edges
        if len(edge) > self.k_size:
            edge = edge[:self.k_size]
            
        while len(edge) < self.k_size:
            edge.append(-1)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)
        
        edge_padding_mask = torch.zeros((1, 1, self.k_size), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == -1, one, edge_padding_mask)
        edge_inputs = torch.where(edge_inputs == -1, 0, edge_inputs)

        # Note: model expects 7 args usually
        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, edge_mask # Using edge_mask as utility_mask for now
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
        return next_position, action_index

    # ... save/load buffer ...
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
        self.total_semantic_gain = 0
        max_steps = 128 
        
        for i in range(max_steps):
            # Single Agent: No loop over robots
            observations = self.get_observations()
            self.save_observations(observations)
            
            next_position, action_index = self.select_node(observations)
            self.save_action(action_index)
            
            new_semantics_count, dist = self.env.step(next_position)
            
            # Update graph (using safe map internal to env)
            self.env.update_graph()

            reward_explore = self.env.calculate_reward(dist)
            reward_semantic = new_semantics_count * 5.0
            total_reward = reward_explore + reward_semantic
            self.total_semantic_gain += new_semantics_count
            
            done = self.env.check_done()
            self.save_reward_done(total_reward, done)
            
            observations = self.get_observations() # Next state
            self.save_next_observations(observations)

            if done:
                break

        # Save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['total_steps'] = i
        self.perf_metrics['semantic_gain'] = self.total_semantic_gain

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
            # assert cnt <= K_SIZE
        return bias_matrix
