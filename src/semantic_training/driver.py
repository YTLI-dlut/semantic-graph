import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import time
import copy
import random

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *
from gpu_manager import GPUMemoryManager

# 忽略一些不必要的警告
import warnings
warnings.filterwarnings('ignore')

def writeToTensorBoard(writer, tensorboardData, curr_episode):
    tensorboardData = np.array(tensorboardData)
    avg_data = np.nanmean(tensorboardData, axis=0)
    
    # 解包数据 (注意顺序必须和 run_update_step 返回值严格一致)
    reward, loss_q, loss_pi, loss_alpha, entropy, alpha, q_val, \
    q_diff, q_std, \
    logp_std, prob_max_mean, policy_grad_norm, q1_grad_norm, target_q_mean, \
    perf_reward, perf_dist, perf_success = avg_data

    # --- 原有指标 ---
    writer.add_scalar('Train/Reward', reward, curr_episode)
    writer.add_scalar('Train/Loss_Q', loss_q, curr_episode)
    writer.add_scalar('Train/Loss_Policy', loss_pi, curr_episode)
    writer.add_scalar('Train/Loss_Alpha', loss_alpha, curr_episode)
    writer.add_scalar('Train/Entropy', entropy, curr_episode)
    writer.add_scalar('Train/Alpha', alpha, curr_episode)
    writer.add_scalar('Train/Q_Value', q_val, curr_episode)
    
    # --- [ICU 监控指标] ---
    
    # 1. QNet 健康度
    writer.add_scalar('Debug_Q/Q_Diff', q_diff, curr_episode)        # 区分度 (越高越好，太低说明Reward太小或特征无区分)
    writer.add_scalar('Debug_Q/Q_Std', q_std, curr_episode)          # Q值标准差
    writer.add_scalar('Debug_Q/Target_Q_Mean', target_q_mean, curr_episode) # 学习目标均值 (太小则需放大Reward)
    writer.add_scalar('Debug_Q/Q1_Grad_Norm', q1_grad_norm, curr_episode)   # 梯度 (不能为0)

    # 2. Policy 健康度
    writer.add_scalar('Debug_Policy/LogP_Std', logp_std, curr_episode)      # Logits离散度 (太小说明tanh限幅或网络输出平坦)
    writer.add_scalar('Debug_Policy/Prob_Max', prob_max_mean, curr_episode) # 最大置信度 (0.02是瞎猜，>0.5是确信)
    writer.add_scalar('Debug_Policy/Grad_Norm', policy_grad_norm, curr_episode) # 梯度 (不能为0)

    # --- 性能指标 ---
    writer.add_scalar('Perf/SemanticGain', perf_reward, curr_episode)
    writer.add_scalar('Perf/TravelDist', perf_dist, curr_episode)
    writer.add_scalar('Perf/SuccessRate', perf_success, curr_episode)

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
        print("Dataset Configuration Validated.")

    # Tensorboard
    writer = SummaryWriter(train_path)

    # GPU Manager
    gpu_manager = GPUMemoryManager()
    device_map = gpu_manager.suggest_device_map()
    print(f"Device Map: {device_map}")

    # Device (Primary)
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print("Warning: CUDA requested but not available! Falling back to CPU.")
        device = torch.device('cpu')
    
    # --- SAC Components Initialization ---
    input_dim = INPUT_DIM
    
    # 1. Initialize Networks
    global_policy_net = PolicyNet(input_dim, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(input_dim, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(input_dim, EMBEDDING_DIM).to(device)
    
    global_target_q_net1 = QNet(input_dim, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(input_dim, EMBEDDING_DIM).to(device)
    
    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # 2. DataParallel Wrappers
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)
    
    # Entropy Parameter
    target_entropy = 0.1 * (-np.log(1.0 / (K_SIZE * NUM_HEADING_CANDIDATES)))
    
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True
    
    # Optimizers
    policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    q_optimizer1 = optim.Adam(global_q_net1.parameters(), lr=LR)
    q_optimizer2 = optim.Adam(global_q_net2.parameters(), lr=LR)
    alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # Load Model
    curr_episode = 0
    target_q_update_counter = 1 

    if LOAD_MODEL and LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading checkpoint from {LOAD_MODEL_PATH}...")
        checkpoint = torch.load(LOAD_MODEL_PATH)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        q_optimizer1.load_state_dict(checkpoint['q_optimizer1'])
        q_optimizer2.load_state_dict(checkpoint['q_optimizer2'])
        alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        try:
            start_episode = int(LOAD_MODEL_PATH.split('_')[-1].split('.')[0])
            curr_episode = start_episode
            print(f"Resuming from episode {curr_episode}")
        except:
            print("Could not parse episode number, starting from 0")

    # Ray Init
    if not ray.is_initialized():
        ray.init()
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    
    # Initial weights for workers
    weights_set = []
    weights_set.append({k: v.cpu() for k, v in global_policy_net.state_dict().items()})
    weights_set.append({k: v.cpu() for k, v in global_q_net1.state_dict().items()})

    # Experience Buffer
    experience_buffer = [] 
    for _ in range(30): 
        experience_buffer.append([])

    # Metrics containers
    training_data = [] 
    max_reward = -float('inf')

    print(f"Starting SAC training with {NUM_META_AGENT} meta agents (Asynchronous Mode)...")
    
    try:
        current_dataset_path = None
        
        # Launch Initial Jobs
        job_list = []
        for i, meta_agent in enumerate(meta_agents):
            if curr_episode < DATASET_EASY_EPOCHS:
                current_dataset_path = DATASET_EASY_PATH
            elif DATASET_MEDIUM_START_EPOCH <= curr_episode < DATASET_MEDIUM_END_EPOCH:
                current_dataset_path = DATASET_MEDIUM_PATH
            else:
                current_dataset_path = DATASET_HARD_PATH
            
            job_list.append(meta_agent.job.remote(weights_set, curr_episode, gifs_path, current_dataset_path))
            curr_episode += 1

        while True:
            # 1. Asynchronous Wait
            done_id, job_list = ray.wait(job_list, num_returns=1)
            done_jobs = ray.get(done_id)
            
            # 2. Process Result
            for job in done_jobs:
                job_results, metrics, info = job
                
                for i in range(len(job_results)):
                    experience_buffer[i] += job_results[i]
                
                agent_id = info['id']
                
                if curr_episode < DATASET_EASY_EPOCHS:
                    new_dataset_path = DATASET_EASY_PATH
                elif DATASET_MEDIUM_START_EPOCH <= curr_episode < DATASET_MEDIUM_END_EPOCH:
                    new_dataset_path = DATASET_MEDIUM_PATH
                else:
                    new_dataset_path = DATASET_HARD_PATH
                
                job_list.append(meta_agents[agent_id].job.remote(weights_set, curr_episode, gifs_path, new_dataset_path))
                curr_episode += 1

            # 3. Training Trigger
            buffer_size = len(experience_buffer[0])
            if buffer_size > REPLAY_SIZE:
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]
                buffer_size = len(experience_buffer[0])

            if buffer_size >= MINIMUM_BUFFER_SIZE:
                if curr_episode % GPU_MONITOR_INTERVAL == 0:
                    gpu_manager.check_and_balance()

                # Define Update Step with OOM Protection
                @GPUMemoryManager.safe_execution
                def run_update_step():
                    nonlocal target_q_update_counter
                    
                    indices = np.random.choice(buffer_size, BATCH_SIZE, replace=False)
                    
                    # Load batch
                    b_node = torch.stack([experience_buffer[0][j] for j in indices]).to(device)
                    b_edge = torch.stack([experience_buffer[1][j] for j in indices]).to(device)
                    b_curr = torch.stack([experience_buffer[2][j] for j in indices]).to(device)
                    b_node_pad = torch.stack([experience_buffer[3][j] for j in indices]).to(device)
                    b_edge_pad = torch.stack([experience_buffer[4][j] for j in indices]).to(device)
                    b_edge_mask = torch.stack([experience_buffer[5][j] for j in indices]).to(device)
                    b_util = torch.stack([experience_buffer[6][j] for j in indices]).to(device)
                    b_best_headings = torch.stack([experience_buffer[7][j] for j in indices]).to(device)

                    b_action = torch.stack([experience_buffer[8][j] for j in indices]).to(device)
                    b_reward = torch.stack([experience_buffer[10][j] for j in indices]).to(device)
                    b_done = torch.stack([experience_buffer[11][j] for j in indices]).to(device)

                    b_next_node = torch.stack([experience_buffer[12][j] for j in indices]).to(device)
                    b_next_edge = torch.stack([experience_buffer[13][j] for j in indices]).to(device)
                    b_next_curr = torch.stack([experience_buffer[14][j] for j in indices]).to(device)
                    b_next_node_pad = torch.stack([experience_buffer[15][j] for j in indices]).to(device)
                    b_next_edge_pad = torch.stack([experience_buffer[16][j] for j in indices]).to(device)
                    b_next_edge_mask = torch.stack([experience_buffer[17][j] for j in indices]).to(device)
                    b_next_util = torch.stack([experience_buffer[18][j] for j in indices]).to(device)
                    b_next_best_headings = torch.stack([experience_buffer[19][j] for j in indices]).to(device)
                    
                    # =================================================================================
                    # [熔断机制] 坏数据检查与跳过
                    # =================================================================================
                    
                    # 1. 检查当前状态 (Current State)
                    curr_mask_flat = b_edge_pad.view(b_edge_pad.size(0), -1)
                    k_size_val = curr_mask_flat.size(1)
                    
                    bad_indices_curr = (curr_mask_flat.sum(dim=1) == k_size_val).nonzero(as_tuple=True)[0]
                    
                    # 2. 检查下一状态 (Next State)
                    next_mask_flat = b_next_edge_pad.view(b_next_edge_pad.size(0), -1)
                    bad_indices_next = (next_mask_flat.sum(dim=1) == k_size_val).nonzero(as_tuple=True)[0]
                    
                    # 3. 如果发现坏数据，打印信息并跳过
                    if len(bad_indices_curr) > 0 or len(bad_indices_next) > 0:
                        print(f"\n{'='*20} [CRITICAL ERROR DETECTED] {'='*20}")
                        print(f"Timestamp: Step {target_q_update_counter}")
                        
                        if len(bad_indices_curr) > 0:
                            idx = bad_indices_curr[0].item()
                            print(f"[Current State BAD] Found {len(bad_indices_curr)} samples. First Index: {idx}")
                            # 可选：打印更详细的信息以便调试
                            # print(f"  -> Edge Inputs (Raw): {b_edge[idx].squeeze().cpu().numpy().tolist()}")
                            
                        if len(bad_indices_next) > 0:
                            idx = bad_indices_next[0].item()
                            print(f"[Next State BAD] Found {len(bad_indices_next)} samples. First Index: {idx}")
                            
                        print(f"{'='*60}\n")
                        print("Skipping bad batch to prevent model collapse (NaN gradients)...")
                        
                        # 返回 None，外层循环会跳过 append
                        return None 

                    # =================================================================================
                    # Step 1: Policy Update
                    # =================================================================================
                    
                    with torch.no_grad():
                        q_values1, _ = dp_q_net1(b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util, neighbor_best_headings=b_best_headings)
                        q_values2, _ = dp_q_net2(b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util, neighbor_best_headings=b_best_headings)
                        q_values = torch.min(q_values1, q_values2)

                        # [DIAGNOSTIC] Q Statistics
                        q_std = q_values.std(dim=1).mean().item()
                        q_diff = (q_values.max(dim=1).values - q_values.min(dim=1).values).mean().item()

                    # Get Logits and LogP
                    logp, _ = dp_policy(b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util, neighbor_best_headings=b_best_headings)
                    
                    # [DIAGNOSTIC] Policy Statistics
                    logp_std = logp.std(dim=1).mean().item()
                    probs = logp.exp()
                    prob_max_mean = probs.max(dim=1).values.mean().item()
                    
                    entropy_term = log_alpha.exp().detach() * logp
                    policy_loss = torch.sum(probs * (entropy_term - q_values.detach()), dim=1).mean()

                    policy_optimizer.zero_grad()
                    
                    # 额外的 NaN 检查 (双重保险)
                    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                        print("[Warning] Policy Loss is NaN/Inf! Skipping step.")
                        return None
                        
                    policy_loss.backward()
                    
                    # [DIAGNOSTIC] Policy Gradient Norm
                    policy_grad_norm = 0.0
                    for p in global_policy_net.parameters():
                        if p.grad is not None:
                            policy_grad_norm += p.grad.data.norm(2).item()
                            
                    torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100, norm_type=2)
                    policy_optimizer.step()

                    # =================================================================================
                    # Step 2: Critic Update
                    # =================================================================================
                    
                    with torch.no_grad():
                        next_logp, _ = dp_policy(b_next_node, b_next_edge, b_next_curr, b_next_node_pad, b_next_edge_pad, b_next_edge_mask, b_next_util, neighbor_best_headings=b_next_best_headings)
                        
                        next_q_values1, _ = dp_target_q_net1(b_next_node, b_next_edge, b_next_curr, b_next_node_pad, b_next_edge_pad, b_next_edge_mask, b_next_util, neighbor_best_headings=b_next_best_headings)
                        next_q_values2, _ = dp_target_q_net2(b_next_node, b_next_edge, b_next_curr, b_next_node_pad, b_next_edge_pad, b_next_edge_mask, b_next_util, neighbor_best_headings=b_next_best_headings)
                        
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        
                        next_probs = next_logp.exp()
                        v_element_wise = next_probs * (next_q_values - log_alpha.exp() * next_logp)
                        value_prime_batch = torch.sum(v_element_wise, dim=1, keepdim=True)
                        
                        target_q_batch = b_reward + GAMMA * (1 - b_done) * value_prime_batch
                        
                        # [DIAGNOSTIC] Target Q Mean
                        target_q_mean = target_q_batch.mean().item()

                    mse_loss = nn.MSELoss()
                    
                    # Q1 Update
                    q_values1_pred, _ = dp_q_net1(b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util, neighbor_best_headings=b_best_headings)
                    q1 = torch.gather(q_values1_pred, 1, b_action.squeeze(-1))
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    q_optimizer1.zero_grad()
                    q1_loss.backward()
                    
                    # [DIAGNOSTIC] Q1 Gradient Norm
                    q1_grad_norm = 0.0
                    for p in global_q_net1.parameters():
                        if p.grad is not None:
                            q1_grad_norm += p.grad.data.norm(2).item()
                            
                    torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000, norm_type=2)
                    q_optimizer1.step()
                    
                    # Q2 Update
                    q_values2_pred, _ = dp_q_net2(b_node, b_edge, b_curr, b_node_pad, b_edge_pad, b_edge_mask, b_util, neighbor_best_headings=b_best_headings)
                    q2 = torch.gather(q_values2_pred, 1, b_action.squeeze(-1))
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    q_optimizer2.zero_grad()
                    q2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000, norm_type=2)
                    q_optimizer2.step()

                    # =================================================================================
                    # Step 3: Alpha Update
                    # =================================================================================
                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + target_entropy)).mean()
                    
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    # =================================================================================
                    # Step 4: Hard Update
                    # =================================================================================
                    if target_q_update_counter > 64:
                        target_q_update_counter = 1
                        global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                        global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                        global_target_q_net1.eval()
                        global_target_q_net2.eval()
                    else:
                        target_q_update_counter += 1
                    
                    return q1_loss.item(), policy_loss.item(), alpha_loss.item(), \
                           entropy.mean().item(), log_alpha.exp().item(), q1.mean().item(), \
                           q_diff, q_std, \
                           logp_std, prob_max_mean, policy_grad_norm, q1_grad_norm, target_q_mean

                # Execute Training Steps
                updates_per_episode = 8
                step_results = []
                for _ in range(updates_per_episode):
                    res = run_update_step()
                    if res:
                        step_results.append(res)
                
                if step_results:
                    avg_res = np.mean(step_results, axis=0) 
                    
                    current_perf = [
                        metrics.get("semantic_gain", 0), 
                        metrics.get("travel_dist", 0), 
                        metrics.get("success_rate", 0)
                    ]
                    log_entry = [current_perf[0], *avg_res, *current_perf]
                    training_data.append(log_entry)

                # Sync Weights for Next Jobs
                weights_set = []
                weights_set.append({k: v.cpu() for k, v in global_policy_net.state_dict().items()})
                weights_set.append({k: v.cpu() for k, v in global_q_net1.state_dict().items()})

            # 4. Logging & Saving
            if curr_episode % SUMMARY_WINDOW == 0 and len(training_data) > 0:
                writeToTensorBoard(writer, training_data, curr_episode)
                
                avg_gain = np.mean([x[-3] for x in training_data])
                if avg_gain >= max_reward:
                    max_reward = avg_gain
                    print(f"[{time.strftime('%H:%M:%S')}] New Best Reward: {max_reward:.2f}. Saving Best Model...")
                    save_dict = {
                        'policy_model': global_policy_net.state_dict(),
                        'q_net1_model': global_q_net1.state_dict(),
                        'q_net2_model': global_q_net2.state_dict(),
                        'log_alpha': log_alpha,
                        'policy_optimizer': policy_optimizer.state_dict(),
                        'q_optimizer1': q_optimizer1.state_dict(),
                        'q_optimizer2': q_optimizer2.state_dict(),
                        'log_alpha_optimizer': alpha_optimizer.state_dict(),
                        'episode': curr_episode
                    }
                    torch.save(save_dict, f'{model_path}/best_checkpoint.pth')
                
                print(f"[{time.strftime('%H:%M:%S')}] Episode {curr_episode} | Avg Gain: {avg_gain:.2f}")
                training_data = [] 

            if curr_episode % 32 == 0:
                save_dict = {
                    'policy_model': global_policy_net.state_dict(),
                    'q_net1_model': global_q_net1.state_dict(),
                    'q_net2_model': global_q_net2.state_dict(),
                    'log_alpha': log_alpha,
                    'policy_optimizer': policy_optimizer.state_dict(),
                    'q_optimizer1': q_optimizer1.state_dict(),
                    'q_optimizer2': q_optimizer2.state_dict(),
                    'log_alpha_optimizer': alpha_optimizer.state_dict(),
                    'episode': curr_episode
                }
                torch.save(save_dict, f'{model_path}/checkpoint.pth')

    except KeyboardInterrupt:
        print("Stopping training...")
        ray.shutdown()

if __name__ == '__main__':
    main()