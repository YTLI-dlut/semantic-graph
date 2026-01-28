
import os
import imageio.v2 as imageio
import glob
import numpy as np
import torch
import shutil
from env import Env
from model import PolicyNet, QNet
from parameter import *

def make_gif(path, duration, filename="agent_movement.gif"):
    images = []
    # Only select step_*.png files to avoid mixing with ground_truth.png which has different dimensions
    filenames = sorted(glob.glob(os.path.join(path, 'step_*.png')))
    for filename_png in filenames:
        images.append(imageio.imread(filename_png))
    
    if not os.path.exists('gifs'):
        os.makedirs('gifs')
        
    output_path = os.path.join('gifs', filename)
    if len(images) > 0:
        imageio.mimsave(output_path, images, duration=duration)
        print(f"GIF saved to {output_path}")
    else:
        print("No images found to create GIF.")

def main():
    # Setup
    global_step = 0
    save_path = f'temp_vis_frames'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    # Initialize Env
    print("Initializing Environment...")
    env = Env(map_index=0, plot=True)
    
    # Initialize Model (Random Weights for now, or load if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = INPUT_DIM
        
    policy_net = PolicyNet(input_dim, EMBEDDING_DIM).to(device)
    
    # Try to load latest model
    model_dirs = sorted(glob.glob('model_save/semantic_*'))
    if len(model_dirs) > 0:
        latest_dir = model_dirs[-1]
        print(f"Found latest model dir: {latest_dir}")
        # Sort by episode number (policy_100.pth)
        try:
            policies = sorted(glob.glob(os.path.join(latest_dir, 'policy_*.pth')), 
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(policies) > 0:
                latest_policy = policies[-1]
                print(f"Loading model: {latest_policy}")
                policy_net.load_state_dict(torch.load(latest_policy, map_location=device))
            else:
                print("No policy files found in latest dir.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No model_save directories found. Using random weights.")
    
    print("Starting Simulation...")
    max_steps = 128
    
    # Simplified Worker-like loop
    from worker import Worker
    # Use greedy=False to urge random exploration with untrained network
    worker = Worker(0, policy_net, None, global_step, device=device, save_image=True, greedy=False)
    worker.env = env # Use our env instance
    
    # Run loop manually to control plotting
    print("Starting Simulation Loop...")
    
    episode_attempt = 0
    while True:
        episode_attempt += 1
        print(f"--- Attempt {episode_attempt} ---")
        
        # Reset Env (Re-init)
        env = Env(map_index=0, plot=True)
        worker.env = env
        
        print(f"Initial Frontiers: {len(env.frontiers)}")
        
        step_count = 0
        done = False
        
        # Unique path for this attempt
        current_save_path = os.path.join(save_path, f"attempt_{episode_attempt}")
        if os.path.exists(current_save_path):
            shutil.rmtree(current_save_path)
        os.makedirs(current_save_path)
        
        # Save GT for comparison
        env.save_ground_truth(current_save_path)
        
        for i in range(max_steps):
            print(f"Step {i}/{max_steps}, Frontiers: {len(env.frontiers)}")
            
            # Get Obs
            observations = worker.get_observations()
            
            # Select Action
            # Pass curr_episode for exploration strategy (e.g. 99999 for model prediction if trained)
            # using episode_attempt as proxy or just large number if we want model output
            next_position, action_index, target_orientation, orientation_idx = worker.select_node(observations, 99999)
            
            # Step Env
            env.step(next_position, target_orientation)
            
            # Update Graph
            env.update_graph()
            
            # Plot
            ori_deg = np.degrees(target_orientation)
            ori_idx = orientation_idx.item()
            info_text = f"Step: {i} | Ori: {ori_deg:.1f} (Idx {ori_idx})"
            env.plot_env(f"attempt_{episode_attempt}", current_save_path, i, info_text=info_text)
            step_count += 1
            
            # Check Done
            if env.check_done():
                print("Exploration Completed!")
                break
        
        if step_count > 5:
            print("Captured a good episode!")
            print("Simulation finished. Generating GIF...")
            print(f"Checking files in {current_save_path}: {len(os.listdir(current_save_path))} files")
            make_gif(current_save_path, 0.2)
            break
        else:
            print("Episode too short, retrying...")
    
    # Cleanup
    # shutil.rmtree(save_path)

if __name__ == '__main__':
    main()
