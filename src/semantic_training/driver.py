
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import time
import copy

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *
from gpu_manager import GPUMemoryManager

def main():
    # Setup paths
    init_dirs()
    print(f"Model Path: {model_path}")
    print(f"Log Path: {train_path}")
    print(f"GIFs Path: {gifs_path}")

    # Validate Dataset Configuration
    if train_mode:
        print("Validating Dataset Configuration...")
        if not os.path.exists(DATASET_EASY_PATH):
            raise FileNotFoundError(f"Easy dataset not found at {DATASET_EASY_PATH}")
        if not os.path.exists(DATASET_MEDIUM_PATH):
            raise FileNotFoundError(f"Medium dataset not found at {DATASET_MEDIUM_PATH}")
        if not os.path.exists(DATASET_HARD_PATH):
            raise FileNotFoundError(f"Hard dataset not found at {DATASET_HARD_PATH}")
        
        # Check Epoch Logic
        if DATASET_EASY_EPOCHS != DATASET_MEDIUM_START_EPOCH:
            print("Warning: Gap or overlap between Easy and Medium epochs.")
        if DATASET_MEDIUM_END_EPOCH != DATASET_HARD_START_EPOCH:
            print("Warning: Gap or overlap between Medium and Hard epochs.")
        
        print("Dataset Configuration Validated.")

    # Tensorboard
    writer = SummaryWriter(train_path)

    # GPU Manager
    gpu_manager = GPUMemoryManager()
    device_map = gpu_manager.suggest_device_map()
    print(f"Device Map: {device_map}")

    # Device (Primary)
    device = torch.device(device_map['q1'])
    if USE_GPU:
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    # --- SAC Components Initialization ---
    
    # 1. Global Networks (Actor & Twin Critics)
    # Distribute networks according to device map
    input_dim = INPUT_DIM
    global_policy_net = PolicyNet(input_dim, EMBEDDING_DIM).to(device_map['policy'])
    global_q_net1 = QNet(input_dim, EMBEDDING_DIM).to(device_map['q1'])
    global_q_net2 = QNet(input_dim, EMBEDDING_DIM).to(device_map['q2'])
    
    # 2. Target Networks (Twin Critics)
    global_target_q_net1 = QNet(input_dim, EMBEDDING_DIM).to(device_map['q1'])
    global_target_q_net2 = QNet(input_dim, EMBEDDING_DIM).to(device_map['q2'])
    
    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()
    
    # 3. Entropy Parameter (Alpha)
    # Target Entropy adjusted for Joint Action Space (K_SIZE * NUM_HEADING_CANDIDATES)
    target_entropy = -np.log(1.0 / (K_SIZE * NUM_HEADING_CANDIDATES)) * TARGET_ENTROPY_SCALE
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    log_alpha.data[:] = np.log(ALPHA)
    
    # Optimizers
    policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    q_optimizer1 = optim.Adam(global_q_net1.parameters(), lr=LR)
    q_optimizer2 = optim.Adam(global_q_net2.parameters(), lr=LR)
    alpha_optimizer = optim.Adam([log_alpha], lr=LR)

    # Load Model
    curr_episode = 0
    if LOAD_MODEL and LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading checkpoint from {LOAD_MODEL_PATH}...")
        checkpoint = torch.load(LOAD_MODEL_PATH)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha.data = checkpoint['log_alpha'].data
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        q_optimizer1.load_state_dict(checkpoint['q_optimizer1'])
        q_optimizer2.load_state_dict(checkpoint['q_optimizer2'])
        alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        try:
            start_episode = int(LOAD_MODEL_PATH.split('_')[-1].split('.')[0])
            curr_episode = start_episode
            print(f"Resuming from episode {curr_episode}")
        except ValueError:
            print("Could not parse episode number from filename, starting from 0")

    # Ray
    ray.init()
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    
    # Initial weights for workers
    weights_set = []
    weights_set.append(global_policy_net.state_dict())
    weights_set.append(global_q_net1.state_dict())

    # Training Loop
    # Experience Replay Buffer
    experience_buffer = [] 
    for _ in range(30): # Reserve enough slots
        experience_buffer.append([])
        
    print(f"Starting SAC training with {NUM_META_AGENT} meta agents...")
    
    try:
        current_dataset_path = None
        while True:
            # Determine Dataset
            if curr_episode < DATASET_EASY_EPOCHS:
                new_dataset_path = DATASET_EASY_PATH
                stage_name = "EASY"
            elif DATASET_MEDIUM_START_EPOCH <= curr_episode < DATASET_MEDIUM_END_EPOCH:
                new_dataset_path = DATASET_MEDIUM_PATH
                stage_name = "MEDIUM"
            else:
                new_dataset_path = DATASET_HARD_PATH
                stage_name = "HARD"

            if new_dataset_path != current_dataset_path:
                print(f"[{time.strftime('%H:%M:%S')}] Switching to {stage_name} dataset: {new_dataset_path}")
                current_dataset_path = new_dataset_path

            # Launch jobs
            job_list = []
            for i, meta_agent in enumerate(meta_agents):
                job_list.append(meta_agent.job.remote(weights_set, curr_episode + i, gifs_path, current_dataset_path))
            
            # Wait for results
            done_id, job_list = ray.wait(job_list, num_returns=NUM_META_AGENT)
            done_results = ray.get(done_id)
            
            # Metrics
            metric_list = []
            
            # Process results & Add to Buffer
            for result in done_results:
                job_results, metrics, info = result
                metric_list.append(metrics)
                
                length = len(job_results[0])
                
            for i in range(length):
                # Current State (索引0-7)
                experience_buffer[0].append(job_results[0][i])  # node_inputs
                experience_buffer[1].append(job_results[1][i])  # edge_inputs
                experience_buffer[2].append(job_results[2][i])  # current_index
                experience_buffer[3].append(job_results[3][i])  # node_padding_mask
                experience_buffer[4].append(job_results[4][i])  # edge_padding_mask
                experience_buffer[5].append(job_results[5][i])  # edge_mask
                experience_buffer[6].append(job_results[6][i])  # utility_mask
                experience_buffer[7].append(job_results[7][i])  # neighbor_best_headings

                # Action (索引8-9)
                experience_buffer[8].append(job_results[8][i])   # action_index
                experience_buffer[9].append(job_results[9][i])   # orientation_idx

                # Reward and Done (索引10-11)
                experience_buffer[10].append(job_results[10][i])  # reward
                experience_buffer[11].append(job_results[11][i])  # done

                # Next State (索引12-19)
                experience_buffer[12].append(job_results[12][i])  # next node_inputs
                experience_buffer[13].append(job_results[13][i])  # next edge_inputs
                experience_buffer[14].append(job_results[14][i])  # next current_index
                experience_buffer[15].append(job_results[15][i])  # next node_padding_mask
                experience_buffer[16].append(job_results[16][i])  # next edge_padding_mask
                experience_buffer[17].append(job_results[17][i])  # next edge_mask
                experience_buffer[18].append(job_results[18][i])  # next utility_mask
                experience_buffer[19].append(job_results[19][i])  # next neighbor_best_headings

            # Trim Buffer
            if len(experience_buffer[0]) > REPLAY_SIZE:
                for i in range(len(experience_buffer)):
                    if len(experience_buffer[i]) > 0:
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]
            
            buffer_size = len(experience_buffer[0])
            
            # --- SAC Update Step ---
            cumulative_loss_q = 0
            cumulative_loss_pi = 0
            cumulative_loss_alpha = 0
            cumulative_entropy = 0
            cumulative_q_val = 0
            num_updates = 0
            
            # Check GPU Memory Balance
            if curr_episode % GPU_MONITOR_INTERVAL == 0:
                gpu_manager.check_and_balance()
                gpu_manager.log_status()
            
            if buffer_size >= MINIMUM_BUFFER_SIZE:
                updates_per_episode = 8 
                
                # Define helper for OOM safety
                @GPUMemoryManager.safe_execution
                def run_update_step():
                    indices = np.random.choice(buffer_size, BATCH_SIZE, replace=False)
                    
                    # Load batch to primary device (q1 device) first
                    b_node = torch.stack([experience_buffer[0][j] for j in indices]).to(device)
                    b_edge = torch.stack([experience_buffer[1][j] for j in indices]).to(device)
                    b_curr = torch.stack([experience_buffer[2][j] for j in indices]).to(device)
                    b_node_pad = torch.stack([experience_buffer[3][j] for j in indices]).to(device)
                    b_edge_pad = torch.stack([experience_buffer[4][j] for j in indices]).to(device)
                    b_edge_mask = torch.stack([experience_buffer[5][j] for j in indices]).to(device)
                    b_util = torch.stack([experience_buffer[6][j] for j in indices]).to(device)  # 修正：索引6是当前utility_mask
                    b_best_headings = torch.stack([experience_buffer[7][j] for j in indices]).to(device)  # 修正：索引7是当前neighbor_best_headings

                    b_action = torch.stack([experience_buffer[8][j] for j in indices]).to(device)  # 修正：索引8是action_index
                    b_orientation = torch.stack([experience_buffer[9][j] for j in indices]).to(device)  # 新增：索引9是orientation_idx
                    b_reward = torch.stack([experience_buffer[10][j] for j in indices]).to(device)  # 修正：索引10是reward
                    b_done = torch.stack([experience_buffer[11][j] for j in indices]).to(device)  # 修正：索引11是done

                    b_next_node = torch.stack([experience_buffer[12][j] for j in indices]).to(device)  # 修正：索引12是下一个node_inputs
                    b_next_edge = torch.stack([experience_buffer[13][j] for j in indices]).to(device)  # 修正：索引13是下一个edge_inputs
                    b_next_curr = torch.stack([experience_buffer[14][j] for j in indices]).to(device)  # 修正：索引14是下一个current_index
                    b_next_node_pad = torch.stack([experience_buffer[15][j] for j in indices]).to(device)  # 修正：索引15是下一个node_padding_mask
                    b_next_edge_pad = torch.stack([experience_buffer[16][j] for j in indices]).to(device)  # 修正：索引16是下一个edge_padding_mask
                    b_next_edge_mask = torch.stack([experience_buffer[17][j] for j in indices]).to(device)  # 修正：索引17是下一个edge_mask
                    b_next_util = torch.stack([experience_buffer[18][j] for j in indices]).to(device)  # 修正：索引18是下一个utility_mask
                    b_next_best_headings = torch.stack([experience_buffer[19][j] for j in indices]).to(device)  # 修正：索引19是下一个neighbor_best_headings
                    
                    # Devices
                    dev_p = device_map['policy']
                    dev_q1 = device_map['q1']
                    dev_q2 = device_map['q2']

                    # 1. Critic Update
                    with torch.no_grad():
                        alpha = log_alpha.exp()
                        # Policy on dev_p
                        next_logp_list, _, _ = global_policy_net(
                            b_next_node.to(dev_p), b_next_edge.to(dev_p), b_next_curr.to(dev_p), 
                            b_next_node_pad.to(dev_p), b_next_edge_pad.to(dev_p), b_next_edge_mask.to(dev_p), b_next_util.to(dev_p),
                            neighbor_best_headings=b_next_best_headings.to(dev_p),
                            return_attention_weights=True
                        )
                        next_logp_list = next_logp_list.to(device)
                        next_probs = next_logp_list.exp()
                        next_log_probs = next_logp_list

                        # Target Q1 on dev_q1 (primary)
                        target_q1_val, _ = global_target_q_net1(
                            b_next_node, b_next_edge, b_next_curr, b_next_node_pad, b_next_edge_pad, b_next_edge_mask, b_next_util,
                            neighbor_best_headings=b_next_best_headings
                        )
                        
                        # Target Q2 on dev_q2
                        target_q2_val, _ = global_target_q_net2(
                            b_next_node.to(dev_q2), b_next_edge.to(dev_q2), b_next_curr.to(dev_q2), 
                            b_next_node_pad.to(dev_q2), b_next_edge_pad.to(dev_q2), b_next_edge_mask.to(dev_q2), b_next_util.to(dev_q2),
                            neighbor_best_headings=b_next_best_headings.to(dev_q2)
                        )
                        target_q2_val = target_q2_val.to(device) # Bring back to primary
                        
                        target_q_min = torch.min(target_q1_val, target_q2_val)
                        
                        next_v = (next_probs * (target_q_min - alpha * next_log_probs)).sum(dim=1, keepdim=True)
                        target_q = b_reward + GAMMA * (1 - b_done) * next_v
                    
                    # Current Q1 on dev_q1
                    current_q1_val, _ = global_q_net1(
                        b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util,
                        neighbor_best_headings=b_best_headings
                    )
                    
                    # Current Q2 on dev_q2
                    current_q2_val, _ = global_q_net2(
                        b_node.to(dev_q2), b_edge.to(dev_q2), b_curr.to(dev_q2), 
                        b_node_pad.to(dev_q2), b_edge_pad.to(dev_q2), b_edge_mask.to(dev_q2), b_util.to(dev_q2),
                        neighbor_best_headings=b_best_headings.to(dev_q2)
                    )
                    
                    # Gather Q1
                    current_q1 = torch.gather(current_q1_val, 1, b_action.squeeze(-1)).squeeze(-1)
                    
                    # Gather Q2 (on dev_q2 then move, or move val then gather)
                    # current_q2_val is on dev_q2. b_action needs to be on dev_q2.
                    current_q2 = torch.gather(current_q2_val, 1, b_action.to(dev_q2).squeeze(-1)).squeeze(-1)
                    
                    # Calculate loss on primary device (dev_q1)
                    loss_q = nn.MSELoss()(current_q1, target_q.squeeze(-1)) + nn.MSELoss()(current_q2.to(device), target_q.squeeze(-1))
                    
                    q_optimizer1.zero_grad()
                    q_optimizer2.zero_grad()
                    loss_q.backward()
                    q_optimizer1.step()
                    q_optimizer2.step()
                    
                    # 2. Policy Update
                    logp_list, _, _ = global_policy_net(
                        b_node.to(dev_p), b_edge.to(dev_p), b_curr.to(dev_p), 
                        b_node_pad.to(dev_p), b_edge_pad.to(dev_p), b_edge_mask.to(dev_p), b_util.to(dev_p), 
                        neighbor_best_headings=b_best_headings.to(dev_p),
                        return_attention_weights=True
                    )
                    probs = logp_list.exp()
                    
                    with torch.no_grad():
                        q1_val, _ = global_q_net1(
                            b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util,
                            neighbor_best_headings=b_best_headings
                        )
                        
                        q2_val, _ = global_q_net2(
                            b_node.to(dev_q2), b_edge.to(dev_q2), b_curr.to(dev_q2), 
                            b_node_pad.to(dev_q2), b_edge_pad.to(dev_q2), b_edge_mask.to(dev_q2), b_util.to(dev_q2),
                            neighbor_best_headings=b_best_headings.to(dev_q2)
                        )
                        q2_val = q2_val.to(device)
                        
                        min_q = torch.min(q1_val, q2_val)
                    
                    # Policy Loss on dev_p
                    # Move components to dev_p
                    loss_policy = (probs.to(dev_p) * (alpha.to(dev_p) * logp_list - min_q.to(dev_p))).sum(dim=1).mean()
                    
                    policy_optimizer.zero_grad()
                    loss_policy.backward()
                    policy_optimizer.step()
                    
                    # 3. Alpha Update
                    with torch.no_grad():
                        entropy = -(probs * logp_list).sum(dim=1).mean()
                    
                    # Alpha is on device (primary)
                    loss_alpha = (log_alpha * (entropy.to(device) - target_entropy).detach())
                    
                    alpha_optimizer.zero_grad()
                    loss_alpha.backward()
                    alpha_optimizer.step()
                    
                    # 4. Soft Update
                    for target_param, param in zip(global_target_q_net1.parameters(), global_q_net1.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
                    for target_param, param in zip(global_target_q_net2.parameters(), global_q_net2.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
                    
                    return loss_q.item(), loss_policy.item(), loss_alpha.item(), entropy.item(), current_q1.mean().item()

                for _ in range(updates_per_episode):
                    results = run_update_step()
                    if results:
                        l_q, l_pi, l_alpha, ent, q_v = results
                        cumulative_loss_q += l_q
                        cumulative_loss_pi += l_pi
                        cumulative_loss_alpha += l_alpha
                        cumulative_entropy += ent
                        cumulative_q_val += q_v
                        num_updates += 1

            # Logging
            if num_updates > 0:
                writer.add_scalar('Train/Loss_Q', cumulative_loss_q/num_updates, curr_episode)
                writer.add_scalar('Train/Loss_Policy', cumulative_loss_pi/num_updates, curr_episode)
                writer.add_scalar('Train/Loss_Alpha', cumulative_loss_alpha/num_updates, curr_episode)
                writer.add_scalar('Train/Entropy', cumulative_entropy/num_updates, curr_episode)
                writer.add_scalar('Train/Alpha', log_alpha.exp().item(), curr_episode)
                writer.add_scalar('Train/Q_Value', cumulative_q_val/num_updates, curr_episode)
                
                if curr_episode % 10 == 0:
                    print(f"Loss Q: {cumulative_loss_q/num_updates:.4f}, Pi: {cumulative_loss_pi/num_updates:.4f}, Alpha: {log_alpha.exp().item():.4f}")

            # Sync weights
            weights_set = []
            weights_set.append({k: v.cpu() for k, v in global_policy_net.state_dict().items()})
            weights_set.append({k: v.cpu() for k, v in global_q_net1.state_dict().items()})
            
            # Perf Metrics
            avg_reward = np.mean([m.get("semantic_gain", 0) for m in metric_list])
            avg_dist = np.mean([m.get("travel_dist", 0) for m in metric_list])
            
            print(f"[{time.strftime('%H:%M:%S')}] Episode {curr_episode}: "
                  f"Avg Semantic Gain: {avg_reward:.2f}, Dist: {avg_dist:.2f}")
            
            writer.add_scalar('Perf/SemanticGain', avg_reward, curr_episode)
            writer.add_scalar('Perf/TravelDist', avg_dist, curr_episode)
            writer.add_scalar('Perf/SuccessRate', np.mean([m.get("success_rate", 0) for m in metric_list]), curr_episode)
            
            # Save model
            if curr_episode % 100 == 0:
                save_dict = {
                    'policy_model': global_policy_net.state_dict(),
                    'q_net1_model': global_q_net1.state_dict(),
                    'q_net2_model': global_q_net2.state_dict(),
                    'log_alpha': log_alpha,
                    'policy_optimizer': policy_optimizer.state_dict(),
                    'q_optimizer1': q_optimizer1.state_dict(),
                    'q_optimizer2': q_optimizer2.state_dict(),
                    'alpha_optimizer': alpha_optimizer.state_dict()
                }
                torch.save(save_dict, f'{model_path}/sac_checkpoint_{curr_episode}.pth')
                
            curr_episode += NUM_META_AGENT
            
    except KeyboardInterrupt:
        print("Stopping training...")
        ray.shutdown()

if __name__ == '__main__':
    main()
