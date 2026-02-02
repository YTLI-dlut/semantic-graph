
import copy
import os
import imageio.v2 as imageio
import shutil
import numpy as np
import torch
from env import Env
from parameter import *
import time
from astar_utils import astar
import matplotlib.pyplot as plt

class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False, save_path=None, dataset_path=None):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image
        self.save_path = save_path if save_path else gifs_path
        self.dataset_path = dataset_path if dataset_path else "generated_data"

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, data_dir=self.dataset_path)
        self.local_policy_net = policy_net
        self.local_q_net = q_net
        
        # Exploration State
        self.current_frontier_target = None
        self.current_path = None
        self.target_replan_timer = 0
        self.blacklisted_frontiers = [] # List of (coords, expiry_step)
        self.position_history = [] # For oscillation detection

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(20):
            self.episode_buffer.append([])
            
        self.total_semantic_gain = 0

    def compute_best_heading(self, node_coords, neighbor_indices):
        """
        Compute Top-3 best heading indices (0-35) for each neighbor node.
        Input:
            node_coords: All node coordinates
            neighbor_indices: Neighbor indices for current step [K_SIZE]
        Output:
            neighbor_best_headings: Tensor [1, K, 3] containing 0-35 indices
        """
        k_size = len(neighbor_indices)
        neighbor_best_headings_list = []
        
        # Get all current frontiers
        if len(self.env.frontiers) > 0:
            frontiers = np.array(list(self.env.frontiers))
        else:
            frontiers = np.empty((0, 2))

        num_angles_bin = NUM_ANGLES_BIN
        sensor_range = self.env.sensor_range
        fov = self.env.fov

        for i in range(k_size):
            node_idx = neighbor_indices[i]
            # Handle padding (-1)
            if node_idx == -1:
                neighbor_best_headings_list.append([0] * NUM_HEADING_CANDIDATES)
                continue

            curr_node_pos = node_coords[node_idx]
            
            # Initialize scores
            heading_scores = np.zeros(num_angles_bin)
            
            if len(frontiers) > 0:
                # 1. Vectors from node to frontiers
                diffs = frontiers - curr_node_pos
                dists = np.linalg.norm(diffs, axis=1)
                
                # 2. Filter within sensor range
                valid_mask = dists < sensor_range
                valid_diffs = diffs[valid_mask]
                
                if len(valid_diffs) > 0:
                    # 3. Relative angles (0 ~ 2pi)
                    angles = np.arctan2(valid_diffs[:, 1], valid_diffs[:, 0])
                    angles = (angles + 2*np.pi) % (2*np.pi)
                    
                    # 4. Map to Bins
                    bin_indices = (angles / (2*np.pi) * num_angles_bin).astype(int)
                    bin_indices = np.clip(bin_indices, 0, num_angles_bin - 1)
                    
                    # 5. Histogram
                    np.add.at(heading_scores, bin_indices, 1)
                    
                    # 6. Smoothing (Convolution)
                    window_size = int((fov / (2*np.pi)) * num_angles_bin) // 2
                    if window_size > 0:
                        kernel = np.ones(window_size)
                        heading_scores = np.convolve(np.pad(heading_scores, window_size, mode='wrap'), kernel, mode='same')[window_size:-window_size]

            # 7. Select Top-N
            if np.sum(heading_scores) > 0:
                top_indices = np.argsort(-heading_scores)[:NUM_HEADING_CANDIDATES]
                # If fewer than N candidates (unlikely with smoothing but possible), pad with 0
                if len(top_indices) < NUM_HEADING_CANDIDATES:
                     padding = np.zeros(NUM_HEADING_CANDIDATES - len(top_indices), dtype=int)
                     top_indices = np.concatenate((top_indices, padding))
            else:
                # Fallback: 0, 120, 240 degrees (indices)
                # 36 bins -> 0, 12, 24
                top_indices = np.array([0, 12, 24]) 
                
            neighbor_best_headings_list.append(top_indices)

        return torch.LongTensor(np.array(neighbor_best_headings_list)).unsqueeze(0).to(self.device)

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
        
        node_utility = node_utility.reshape((n_nodes, 1))

        # New Features: Entropy & Unconfirmed Object Vector
        entropy_feats, vector_feats = self.env.get_node_features(self.env.node_coords)

        node_inputs = np.concatenate((node_coords, node_utility, guidepost, entropy_feats, vector_feats), axis=1)
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
        
        # Utility mask (Attention matrix)
        utility_mask = edge_mask 

        # Padding edges
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        edge = edge_inputs[current_node_index]
        # Clip to K_SIZE if data has more edges
        if len(edge) > self.k_size:
            edge = edge[:self.k_size]
            
        while len(edge) < self.k_size:
            edge.append(-1)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # New: Compute Best Headings
        neighbor_best_headings = self.compute_best_heading(self.env.node_coords, edge)
        
        edge_padding_mask = torch.zeros((1, 1, self.k_size), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == -1, one, edge_padding_mask)
        edge_inputs = torch.where(edge_inputs == -1, 0, edge_inputs)

        # Note: model expects 7 args usually
        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, edge_mask, neighbor_best_headings 
        return observations

    def select_node(self, observations, curr_episode):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask, neighbor_best_headings = observations
        
        # Random Exploration Phase
        if curr_episode < RANDOM_EXPLORE_EPOCHS:
            action_index = None
            orientation_idx = None

            # --- A* Frontier Exploration Logic ---
            if USE_ASTAR_EXPLORATION:
                # 0. Manage Blacklist
                current_step = self.global_step if hasattr(self, 'global_step') else time.time()
                if not hasattr(self, 'internal_step_counter'):
                    self.internal_step_counter = 0
                self.internal_step_counter += 1
                
                self.blacklisted_frontiers = [x for x in self.blacklisted_frontiers if x[1] > self.internal_step_counter]

                # Oscillation Detection
                self.position_history.append(self.env.robot_position)
                if len(self.position_history) > 10:
                    self.position_history.pop(0)
                
                # Check for A-B-A pattern
                is_oscillating = False
                if len(self.position_history) >= 4:
                    p_curr = self.position_history[-1]
                    p_prev = self.position_history[-2]
                    p_prev2 = self.position_history[-3]
                    p_prev3 = self.position_history[-4]
                    
                    if np.linalg.norm(p_curr - p_prev2) < 2.0 and np.linalg.norm(p_prev - p_prev3) < 2.0:
                         is_oscillating = True
                
                if is_oscillating and self.current_frontier_target is not None:
                    self.blacklisted_frontiers.append((self.current_frontier_target, self.internal_step_counter + 100))
                    self.current_frontier_target = None
                    self.current_path = None
                    self.position_history = [] 

                # 1. Update Target Logic
                dist_to_target = float('inf')
                target_valid = False
                if self.current_frontier_target is not None:
                    dist_to_target = np.linalg.norm(self.env.robot_position - self.current_frontier_target)
                    if len(self.env.frontiers) > 0:
                        dists_to_frontiers = np.linalg.norm(self.env.frontiers - self.current_frontier_target, axis=1)
                        if np.min(dists_to_frontiers) < 5.0: 
                            target_valid = True
                
                if not target_valid:
                    self.current_frontier_target = None
                    self.current_path = None
                
                if self.current_frontier_target is not None and dist_to_target < 15.0:
                    self.blacklisted_frontiers.append((self.current_frontier_target, self.internal_step_counter + 50)) 
                    self.current_frontier_target = None
                    self.current_path = None
                
                # Re-select target
                if self.current_frontier_target is None:
                     if len(self.env.frontiers) > 0:
                        candidate_indices = []
                        for i, f in enumerate(self.env.frontiers):
                            is_blacklisted = False
                            for b_coords, _ in self.blacklisted_frontiers:
                                if np.linalg.norm(f - b_coords) < 5.0:
                                    is_blacklisted = True
                                    break
                            if not is_blacklisted:
                                candidate_indices.append(i)
                        
                        if len(candidate_indices) > 0:
                            candidates = self.env.frontiers[candidate_indices]
                            dists = np.linalg.norm(candidates - self.env.robot_position, axis=1)
                            min_idx = np.argmin(dists)
                            self.current_frontier_target = candidates[min_idx]
                        else:
                            dists = np.linalg.norm(self.env.frontiers - self.env.robot_position, axis=1)
                            min_idx = np.argmin(dists)
                            self.current_frontier_target = self.env.frontiers[min_idx]
                            
                        self.current_path = None 
                     else:
                        self.current_frontier_target = None
                        self.current_path = None
                
                # 2. Plan Path
                if self.current_frontier_target is not None:
                    need_replan = False
                    if self.current_path is None or len(self.current_path) == 0:
                        need_replan = True

                    if need_replan:
                        grid_map = (self.env.robot_belief == 1).astype(int)
                        start_pos = (int(self.env.robot_position[1]), int(self.env.robot_position[0])) 
                        end_pos = (int(self.current_frontier_target[1]), int(self.current_frontier_target[0])) 
                        path = astar(grid_map, start_pos, end_pos)
                        
                        if path is None:
                            self.current_frontier_target = None
                            self.current_path = None
                        else:
                            self.current_path = path

                    # 3. Follow Path
                    if self.current_path is not None and len(self.current_path) > 0:
                        path_arr = np.array(self.current_path) 
                        path_xy = np.fliplr(path_arr) 
                        dists = np.linalg.norm(path_xy - self.env.robot_position, axis=1)
                        min_dist_idx = np.argmin(dists)
                        self.current_path = self.current_path[min_dist_idx:]
                        
                        lookahead_dist = 30.0 
                        target_point = self.current_path[-1] 
                        
                        for i, pt in enumerate(self.current_path):
                             pt_xy = np.array([pt[1], pt[0]])
                             dist = np.linalg.norm(pt_xy - self.env.robot_position)
                             if dist > lookahead_dist:
                                 target_point = pt
                                 break
                        
                        target_xy = np.array([target_point[1], target_point[0]])
                        
                        # 4. Select Action
                        valid_mask = (edge_padding_mask == 0).squeeze()
                        valid_indices = torch.nonzero(valid_mask).flatten()
                        
                        best_score = float('inf')
                        best_idx = None
                        
                        if len(valid_indices) > 0:
                            curr_node_idx_val = current_index.item()
                            for idx in valid_indices:
                                n_global_idx = edge_inputs[0, 0, idx].item()
                                if n_global_idx == -1: continue
                                if n_global_idx == curr_node_idx_val: continue 
                                
                                n_pos = self.env.node_coords[n_global_idx]
                                score = np.linalg.norm(n_pos - target_xy)
                                
                                if score < best_score:
                                    best_score = score
                                    best_idx = idx
                            
                            if best_idx is not None:
                                current_dist_to_lookahead = np.linalg.norm(self.env.robot_position - target_xy)
                                if best_score >= current_dist_to_lookahead:
                                     if current_dist_to_lookahead < 20.0:
                                         if self.current_frontier_target is not None:
                                             self.blacklisted_frontiers.append((self.current_frontier_target, self.internal_step_counter + 50))
                                             self.current_frontier_target = None
                                             self.current_path = None
                                         pass
                                     else:
                                         self.current_path = None

                                # --- New Action Logic ---
                                n_global_idx = edge_inputs[0, 0, best_idx].item()
                                n_pos = self.env.node_coords[n_global_idx]
                                dx = n_pos[0] - self.env.robot_position[0]
                                dy = n_pos[1] - self.env.robot_position[1]
                                angle = np.arctan2(dy, dx) 
                                if angle < 0: angle += 2*np.pi
                                
                                # Find best rank
                                candidates_bins = neighbor_best_headings[0, best_idx] # (3,)
                                candidates_angles = candidates_bins.float() * (2*np.pi / NUM_ANGLES_BIN)
                                
                                # Circular difference
                                diffs = torch.abs(candidates_angles - angle)
                                diffs = torch.min(diffs, 2*np.pi - diffs)
                                rank = torch.argmin(diffs).item()
                                
                                action_index = best_idx * NUM_HEADING_CANDIDATES + rank
                                orientation_idx = candidates_bins[rank]
                                
                                action_index = torch.tensor([action_index]).to(self.device)
                                orientation_idx = orientation_idx.unsqueeze(0).to(self.device)

            if action_index is None:
                # Fallback to Random
                valid_mask = (edge_padding_mask == 0).squeeze() # (k_size)
                valid_indices = torch.nonzero(valid_mask).flatten()
                
                curr_node_idx_val = current_index.item()
                filtered_indices = []
                for idx in valid_indices:
                     n_global_idx = edge_inputs[0, 0, idx].item()
                     if n_global_idx != -1 and n_global_idx != curr_node_idx_val:
                         filtered_indices.append(idx)
                
                if len(filtered_indices) > 0:
                    idx = torch.randint(0, len(filtered_indices), (1,)).item()
                    best_idx = filtered_indices[idx]
                else:
                    if len(valid_indices) > 0:
                         best_idx = valid_indices[0]
                    else:
                         best_idx = torch.tensor(0).to(self.device)
                
                # Random Rank
                rank = torch.randint(0, NUM_HEADING_CANDIDATES, (1,)).item()
                action_index = best_idx * NUM_HEADING_CANDIDATES + rank
                action_index = action_index.unsqueeze(0).to(self.device)
                
                orientation_idx = neighbor_best_headings[0, best_idx, rank].unsqueeze(0)
            
        else:
            # Model Prediction Phase
            with torch.no_grad():
                # PolicyNet now returns logp for (K_SIZE * NUM_HEADING_CANDIDATES) actions
                logp_list, _ = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                                  edge_padding_mask, edge_mask, utility_mask, neighbor_best_headings, self.greedy)
            
            if self.greedy:
                action_flat = torch.argmax(logp_list, dim=1).long()
            else:
                action_flat = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            
            action_index = action_flat
            
            neighbor_idx = action_index.item() // NUM_HEADING_CANDIDATES
            rank = action_index.item() % NUM_HEADING_CANDIDATES
            orientation_idx = neighbor_best_headings[0, neighbor_idx, rank].unsqueeze(0)
        
        neighbor_idx = action_index.item() // NUM_HEADING_CANDIDATES
        next_node_index = edge_inputs[0, 0, neighbor_idx]
        next_position = self.env.node_coords[next_node_index]
        
        # Convert orientation index [0-35] to radians [0, 2pi)
        target_orientation = orientation_idx.item() * (2 * np.pi / NUM_ANGLES_BIN)
        
        return next_position, action_index, target_orientation, orientation_idx

    # ... save/load buffer ...
    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask, neighbor_best_headings = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[4] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[5] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[6] += copy.deepcopy(utility_mask).bool()
        self.episode_buffer[7] += copy.deepcopy(neighbor_best_headings)

    def save_action(self, action_index, orientation_idx):
        self.episode_buffer[8] += action_index.unsqueeze(0).unsqueeze(0)
        self.episode_buffer[9] += orientation_idx.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[10] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[11] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, utility_mask, neighbor_best_headings = observations
        self.episode_buffer[12] += copy.deepcopy(node_inputs)
        self.episode_buffer[13] += copy.deepcopy(edge_inputs)
        self.episode_buffer[14] += copy.deepcopy(current_index)
        self.episode_buffer[15] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[16] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[17] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[18] += copy.deepcopy(utility_mask).bool()
        self.episode_buffer[19] += copy.deepcopy(neighbor_best_headings)

    def run_episode(self, curr_episode):
        done = False
        self.total_semantic_gain = 0
        max_steps = MAX_EPISODE_STEPS
        self.current_frontier_target = None # Reset exploration target
        
        # Log Phase
        if curr_episode < RANDOM_EXPLORE_EPOCHS:
            phase_str = "Random Exploration"
        else:
            phase_str = "Model Prediction"
            
        print(f"Agent {self.metaAgentID} | Episode {curr_episode} | Phase: {phase_str}")

        for i in range(max_steps):
            # Single Agent: No loop over robots
            observations = self.get_observations()
            self.save_observations(observations)
            if torch.all(observations[4]):
                print(f"Agent {self.metaAgentID}: Spawned/Moved to dead end. Ending episode.")
                break
            # # === 在这里插入可视化代码 ===
            # # 取出 node_inputs (observations 的第一个元素)
            # node_inputs_tensor = observations[0] 
            
            # # 为了不每一步都存，可以加个判断，比如每10步存一次，或者只在特定 Episode 存
            # if True: 
            #     debug_path = f'{self.save_path}/debug_episode_{curr_episode}'
            #      # 取出 node_inputs (它是 observations 的第0个元素)
            #     # self.debug_plot_separate(observations[0], i, debug_path)
            #     self.visualize_observations(observations, i, debug_path)
            # # ==========================


            next_position, action_index, target_orientation, orientation_idx = self.select_node(observations, curr_episode)
            self.save_action(action_index, orientation_idx)

            # ================= [DEBUG START: 动作显微镜] =================
            # 解码动作：Action ID = Neighbor_Idx * 3 + Heading_Rank
            # neighbor_idx = action_index.item() // NUM_HEADING_CANDIDATES
            # heading_rank = action_index.item() % NUM_HEADING_CANDIDATES
            
            # # 获取目标点的全局 ID (从 observation 的 edge_inputs 里取)
            # # observations[1] is edge_inputs: (1, 1, K)
            # target_node_global_id = observations[1][0, 0, neighbor_idx].item()
            
            # # 计算这一步想走多远
            # dist_to_target = np.linalg.norm(next_position - self.env.robot_position)
            
            # print(f"\n--- [Step {i}] Action Diagnosis ---")
            # print(f"Raw Action Index : {action_index.item()}")
            # print(f"  |__ Neighbor   : Index {neighbor_idx} (Global Node {target_node_global_id})")
            # print(f"  |__ Heading    : Rank {heading_rank}")
            # print(f"Position Check   :")
            # print(f"  |__ Robot Curr : {self.env.robot_position}")
            # print(f"  |__ Target Pos : {next_position}")
            # print(f"  |__ Distance   : {dist_to_target:.6f}")
            
            # if dist_to_target < 1e-4:
            #     print("⚠️ [ALERT] 智能体选择了原地不动 (Stay)！")
            # elif neighbor_idx == 0:
            #      print("ℹ️ [INFO] 智能体选择了第0个邻居 (通常是自己或最近点)")
            
            # print("-" * 40)
            # ================= [DEBUG END] =================
            
            # Record State for Penalty
            prev_pos = self.env.robot_position.copy()
            prev_ori = self.env.robot_orientation
            
            new_semantics_count, dist = self.env.step(next_position, target_orientation)
            

            reward_explore = self.env.calculate_reward(dist)
            reward_semantic = new_semantics_count * REWARD_CONFIRM
            
            # Penalty Logic
            penalty = 0.0
            # Check if stayed still (Position close AND Orientation close)
            dist_moved = np.linalg.norm(self.env.robot_position - prev_pos)
            ori_diff = abs(self.env.robot_orientation - prev_ori)
            ori_diff = min(ori_diff, 2*np.pi - ori_diff) # Cyclic diff
            
            if dist_moved < 1e-3 and ori_diff < 1e-3:
                penalty = STAY_STILL_PENALTY
                # print(f"Ep {curr_episode} Step {i}: Penalty Triggered! Reward += {penalty}")
            penalty += REWARD_STEP_PENALTY
            done = self.env.check_done()
            reward_done = REWARD_DONE if done else 0.0

            # total_reward = reward_explore + reward_semantic + penalty + reward_done + REWARD_STEP_PENALTY
            total_reward = reward_explore + penalty + reward_done
            total_reward /= 50;
            self.total_semantic_gain += new_semantics_count

            if self.save_image:
                 path = f'{self.save_path}/episode_{curr_episode}'
                 info_text = f"步骤: {i} | 回合: {curr_episode}\n"
                 info_text += f"阶段: {phase_str}\n"
                 info_text += f"总奖励: {total_reward * 50:.2f} (探索: {reward_explore:.2f}, 语义: {reward_semantic:.2f}, 惩罚: {penalty:.2f}, 完成: {reward_done:.2f})\n"
                 info_text += f"已确认: {len(self.env.found_semantics)} | 已发现: {len(self.env.seen_semantics)}\n"
                #  ori_deg = np.degrees(target_orientation)
                #  ori_idx = orientation_idx.item()
                #  info_text += f"动作: 移动->Node{next_position} | 角度->Idx{ori_idx}({ori_deg:.1f}° | {action_index.item()})"
                 # === 修改这一部分 ===
                 neighbor_idx = action_index.item() // NUM_HEADING_CANDIDATES
                 heading_rank = action_index.item() % NUM_HEADING_CANDIDATES
                 target_node_id = observations[1][0, 0, neighbor_idx].item()
                 
                 info_text += f"Action: {action_index.item()} = N[{neighbor_idx}] + H[{heading_rank}]\n"
                 info_text += f"Move: Node {target_node_id} (Dist: {np.linalg.norm(next_position - self.env.robot_position):.2f})"
                 # ===================
                 self.env.plot_env(self.global_step, path, i, info_text=info_text)
            
            self.save_reward_done(total_reward, done)
            
            observations_next = self.get_observations()
            
            # === [熔断检查 2] 下一步是死路？(关键！) ===
            # 如果下一状态全是 Mask，Target Q 计算会 NaN，必须丢弃这一步数据
            if torch.all(observations_next[4]):
                # print(f"Agent {self.metaAgentID}: Next state is dead end. Discarding last transition.")
                
                # [回滚操作]：把刚才存进去的 s, a, r 全部吐出来
                # 我们在本 Step 依次调用了：
                # save_observations (存了 8 个 Tensor: indices 0-7)
                # save_action (存了 2 个 Tensor: indices 8-9)
                # save_reward_done (存了 2 个 Tensor: indices 10-11)
                # 总共 12 个，必须全部 pop 掉，保证 Buffer 长度对齐
                for idx in range(12):
                    self.episode_buffer[idx].pop()
                
                # 遇到死路，直接结束本回合
                break
                
            self.save_next_observations(observations_next)
            if done:
                break

        # Save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['total_steps'] = i
        self.perf_metrics['semantic_gain'] = self.total_semantic_gain
        
        # Detailed Rewards & State Metrics
        self.perf_metrics['reward_explore'] = self.env.last_rewards.get('explore', 0)
        # self.perf_metrics['reward_entropy'] = self.env.last_rewards.get('entropy', 0) # Removed
        self.perf_metrics['reward_semantic'] = self.total_semantic_gain * REWARD_CONFIRM # Approx
        self.perf_metrics['reward_penalty'] = penalty # Last step penalty

        
        # self.perf_metrics['state_entropy'] = np.sum(self.env.entropy_map)
        # Calculate entropy sum from nodes (Approximate global entropy)
        if self.env.node_coords is not None and len(self.env.node_coords) > 0:
            entropy_scores, _ = self.env.get_node_features(self.env.node_coords)
            self.perf_metrics['state_entropy'] = np.sum(entropy_scores)
        else:
            self.perf_metrics['state_entropy'] = 0.0
        self.perf_metrics['state_frontiers'] = len(self.env.frontiers)
        self.perf_metrics['state_confirmed'] = len(self.env.found_semantics)
        self.perf_metrics['state_unconfirmed'] = len(self.env.get_unconfirmed_centers())

        if self.save_image:
             self.make_gif(curr_episode)

    def work(self, currEpisode):
        self.run_episode(currEpisode)

    def make_gif(self, episode):
        # Read all images from the folder
        # Path: gifs/FOLDER_NAME/episode_{episode}_step_{step}.png
        # But wait, env.plot_env saves to f'{path}/step_{step_idx:04d}.png'
        # In runner.py: worker = Worker(..., save_image=True/False)
        # In worker.py: self.env = Env(..., plot=save_image)
        # We need to pass the save path to env.plot_env or handle it here.
        # Currently env.plot_env is called? No, it's not called in run_episode yet!
        # We need to call env.plot_env in run_episode if save_image is True.
        
        path = f'{self.save_path}/episode_{episode}'
        images = []
        if not os.path.exists(path):
            return

        file_names = sorted((fn for fn in os.listdir(path) if fn.endswith('.png')))
        for filename in file_names:
            images.append(imageio.imread(os.path.join(path, filename)))
            
        if len(images) > 0:
            gif_path = f'{self.save_path}/episode_{episode}.gif'
            imageio.mimsave(gif_path, images, duration=0.1)
            print(f"GIF generated successfully: {gif_path}")
            # Optional: Clean up images
            if os.path.exists(path):
                shutil.rmtree(path)


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



    def debug_plot_separate(self, node_inputs, step_idx, path):
            """
            分图绘制 Node Inputs 的指标。
            左图：Entropy (关注熵是否正确计算)
            右图：Utility (关注效用值)
            """
            import matplotlib.pyplot as plt
            import cv2
            
            # --- 1. 数据解析 ---
            if isinstance(node_inputs, torch.Tensor):
                data = node_inputs.cpu().detach().numpy()
            else:
                data = node_inputs
            
            # 去掉 batch 维度，确保是 (N, 7)
            if data.ndim == 4: data = data[0, 0]
            elif data.ndim == 3: data = data[0]

            # --- 2. 准备底图 (Map) ---
            # 复用 Env 的地图逻辑 (白:Free, 黑:Obs, 灰:Unknown)
            h, w = self.env.map_size
            map_img = np.zeros((h, w, 3), dtype=np.uint8)
            map_img[self.env.robot_belief == 255] = [255, 255, 255]
            map_img[self.env.robot_belief == 1] = [0, 0, 0]
            map_img[self.env.robot_belief == 127] = [127, 127, 127]
            
            # Matplotlib 显示图片需要 RGB，OpenCV 默认也是构建的 RGB numpy 数组这里没问题
            
            # --- 3. 创建画布 (1行2列) ---
            fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=100)
            norm_scale = max(h, w)
            
            # 定义要画的两个指标配置
            # (标题, 特征在node_inputs里的索引, 颜色条名称, 文本颜色)
            plots_config = [
                ("Entropy (Is it catching Frontiers?)", 4, "Reds", "darkred"), # index 4 is Entropy
                ("Utility (Is it guiding movement?)", 2, "Blues", "darkblue")  # index 2 is Utility
            ]

            # 获取有效节点 (过滤掉 padding 的 0,0 坐标)
            valid_mask = (data[:, 0] != 0) | (data[:, 1] != 0)
            valid_nodes = data[valid_mask]
            
            # --- 4. 循环绘制两个子图 ---
            for ax, (title, feat_idx, cmap_name, text_color) in zip(axes, plots_config):
                # A. 显示地图背景
                ax.imshow(map_img, origin='upper') # origin='upper' 匹配图像坐标系
                
                # B. 画 Frontiers (作为参考真值，用显眼的红叉)
                if len(self.env.frontiers) > 0:
                    ax.scatter(self.env.frontiers[:, 0], self.env.frontiers[:, 1], 
                            c='red', marker='x', s=30, label='Frontier (Truth)', zorder=2)

                # C. 画 Robot (蓝色圆点)
                ax.scatter(self.env.robot_position[0], self.env.robot_position[1], 
                        c='cyan', s=100, edgecolors='k', label='Robot', zorder=5)

                # D. 画 Nodes (散点图，颜色深浅代表数值大小)
                if len(valid_nodes) > 0:
                    node_x = valid_nodes[:, 0] * norm_scale
                    node_y = valid_nodes[:, 1] * norm_scale
                    values = valid_nodes[:, feat_idx]
                    
                    # 散点图
                    sc = ax.scatter(node_x, node_y, c=values, cmap=cmap_name, 
                                    s=80, edgecolors='black', zorder=3, vmin=0, vmax=1.0)
                    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

                    # E. 标注数值 (只标当前指标)
                    for i in range(len(node_x)):
                        val = values[i]
                        # 只有当数值大于 0.01 时才显示，避免 0 值挤满屏幕
                        if val > 0.01:
                            ax.text(node_x[i]+2, node_y[i]-2, f"{val:.2f}", 
                                    color=text_color, fontsize=9, fontweight='bold', zorder=4)
                
                ax.set_title(f"{title} - Step {step_idx}", fontsize=16)
                ax.legend(loc='upper right')
                # 隐藏坐标轴刻度，只看图
                ax.axis('off')

            plt.tight_layout()
            
            # --- 5. 保存 ---
            if not os.path.exists(path):
                os.makedirs(path)
            save_file = os.path.join(path, f'debug_split_step_{step_idx:04d}.png')
            plt.savefig(save_file)
            plt.close(fig)
            # print(f"Saved: {save_file}")

    def visualize_observations(self, observations, step_idx, path):
            """
            全能可视化：Entropy, Utility, 以及 Graph Structure (Edges & Current Pos)
            observations: get_observations() 返回的 tuple
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            import torch

            # --- 1. 解包数据 (全部转为 Numpy) ---
            # observations 结构: 
            # 0:node_inputs, 1:edge_inputs, 2:current_index, 3:node_padding_mask, 
            # 4:edge_padding_mask, 5:edge_mask, 6:utility_mask, 7:best_headings
            
            def to_np(t):
                if isinstance(t, torch.Tensor):
                    return t.cpu().detach().numpy()
                return t

            node_inputs = to_np(observations[0])   # (1, 1, N, 7) or (1, N, 7)
            local_edges = to_np(observations[1])   # (1, 1, K_SIZE)
            curr_idx_ts = to_np(observations[2])   # (1, 1, 1)
            adj_mask    = to_np(observations[5])   # (1, N, N) or (1, 1, N, N)

            # 处理 Node Data 维度
            if node_inputs.ndim == 4: 
                nodes_data = node_inputs[0, 0]
            elif node_inputs.ndim == 3:
                nodes_data = node_inputs[0]
            else:
                nodes_data = node_inputs

            curr_node_idx = int(curr_idx_ts.flatten()[0])
            
            # 处理 Adjacency Matrix 维度 (修复了这里)
            if adj_mask.ndim == 4: 
                full_adj = adj_mask[0, 0]
            elif adj_mask.ndim == 3:
                full_adj = adj_mask[0]
            else:
                full_adj = adj_mask

            # 处理 Local Edges 维度
            if local_edges.ndim == 3: 
                neighbors = local_edges[0, 0]
            elif local_edges.ndim == 2:
                neighbors = local_edges[0]
            else:
                neighbors = local_edges

            # --- 2. 准备底图 ---
            h, w = self.env.map_size
            map_img = np.zeros((h, w, 3), dtype=np.uint8)
            map_img[self.env.robot_belief == 255] = [255, 255, 255]
            map_img[self.env.robot_belief == 1] = [0, 0, 0]
            map_img[self.env.robot_belief == 127] = [127, 127, 127]
            
            # --- 3. 创建画布 (1行3列) ---
            fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=100)
            norm_scale = max(h, w)
            
            # 过滤有效节点
            valid_mask = (nodes_data[:, 0] != 0) | (nodes_data[:, 1] != 0)
            valid_indices = np.where(valid_mask)[0]
            
            node_coords = nodes_data[:, :2] * norm_scale
            
            # === 图1 & 图2: Entropy 和 Utility ===
            plots_config = [
                (axes[0], "Entropy (Channel 4)", 4, "Reds", "darkred"),
                (axes[1], "Utility (Channel 2)", 2, "Blues", "darkblue")
            ]
            
            for ax, title, feat_idx, cmap, text_c in plots_config:
                ax.imshow(map_img, origin='upper')
                ax.scatter(self.env.frontiers[:, 0], self.env.frontiers[:, 1], c='red', marker='x', s=30, label='Frontiers')
                
                # 画节点
                if len(valid_indices) > 0:
                    sc = ax.scatter(node_coords[valid_indices, 0], node_coords[valid_indices, 1], 
                                    c=nodes_data[valid_indices, feat_idx], cmap=cmap, 
                                    s=60, edgecolors='k', vmin=0, vmax=1.0, zorder=3)
                    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    
                    # 写数值
                    for i in valid_indices:
                        val = nodes_data[i, feat_idx]
                        if val > 0.01:
                            ax.text(node_coords[i, 0]+2, node_coords[i, 1]-2, f"{val:.2f}", 
                                    color=text_c, fontsize=8, fontweight='bold', zorder=4)
                
                # 画机器人当前位置
                rx, ry = self.env.robot_position
                ax.scatter(rx, ry, c='cyan', s=100, edgecolors='k', marker='*', label='Robot Pos', zorder=5)
                ax.set_title(title)
                ax.axis('off')

            # === 图3: Connectivity (Graph Structure) ===
            ax = axes[2]
            ax.imshow(map_img, origin='upper')
            ax.set_title(f"Connectivity & Actions\nCurrent Node: {curr_node_idx}")
            
            # A. 画全图连接 (Edge Mask) - 灰色细线
            # full_adj[i, j] == 0 表示 i 和 j 相连
            N = len(nodes_data)
            for i in valid_indices:
                for j in valid_indices:
                    if i < j: 
                        # 确保是标量比较
                        is_connected = full_adj[i, j]
                        if isinstance(is_connected, np.ndarray):
                            is_connected = is_connected.item()
                        
                        if is_connected == 0: 
                            p1 = node_coords[i]
                            p2 = node_coords[j]
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', alpha=0.3, linewidth=0.5, zorder=1)

            # B. 画当前节点 (高亮)
            if curr_node_idx < N:
                curr_pos = node_coords[curr_node_idx]
                ax.scatter(curr_pos[0], curr_pos[1], c='lime', s=150, edgecolors='k', marker='o', label='Current Node', zorder=4)
                # 机器人连线
                rx, ry = self.env.robot_position
                ax.plot([rx, curr_pos[0]], [ry, curr_pos[1]], 'g--', alpha=0.5)

            # C. 画可选动作 (Edge Inputs) - 蓝色粗线
            for n_idx in neighbors:
                n_idx = int(n_idx)
                if n_idx != -1 and n_idx < N:
                    target_pos = node_coords[n_idx]
                    ax.arrow(curr_pos[0], curr_pos[1], 
                            target_pos[0]-curr_pos[0], target_pos[1]-curr_pos[1], 
                            color='blue', width=0.5, head_width=2, length_includes_head=True, zorder=2)
                    ax.text(target_pos[0], target_pos[1], str(n_idx), color='blue', fontsize=8)

            # D. 画其他节点
            ax.scatter(node_coords[valid_indices, 0], node_coords[valid_indices, 1], c='white', edgecolors='gray', s=30, zorder=2)
            ax.scatter(self.env.frontiers[:, 0], self.env.frontiers[:, 1], c='red', marker='x', s=20, alpha=0.5)
            
            ax.legend(loc='upper right')
            ax.axis('off')

            plt.tight_layout()
            
            # --- 保存 ---
            if not os.path.exists(path):
                os.makedirs(path)
            save_file = os.path.join(path, f'debug_full_step_{step_idx:04d}.png')
            plt.savefig(save_file)
            plt.close(fig)