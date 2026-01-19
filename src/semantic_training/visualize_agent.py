
import os
import imageio
import glob
import numpy as np
import torch
import shutil
from env import Env
from model import PolicyNet, QNet
from parameter import *

def make_gif(path, duration, filename="agent_movement.gif"):
    images = []
    filenames = sorted(glob.glob(os.path.join(path, '*.png')))
    for filename_png in filenames:
        images.append(imageio.imread(filename_png))
    
    if not os.path.exists('gifs'):
        os.makedirs('gifs')
        
    output_path = os.path.join('gifs', filename)
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")

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
    if USE_K_FLAGS:
        input_dim += N_ROBOTS
        
    policy_net = PolicyNet(input_dim, EMBEDDING_DIM).to(device)
    
    # Check for saved model
    # model_path defined in parameter.py
    # if os.path.exists(f'{model_path}/policy_0.pth'):
    #     policy_net.load_state_dict(torch.load(f'{model_path}/policy_0.pth'))
    #     print("Loaded saved model.")
    
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
            next_position, action_index = worker.select_node(observations)
            
            # Step Env
            env.step(next_position)
            
            # Update Graph
            env.update_graph()
            
            # Plot
            env.plot_env(f"attempt_{episode_attempt}", current_save_path, i)
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
