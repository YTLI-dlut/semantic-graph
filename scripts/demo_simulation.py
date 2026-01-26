import os
import sys
import numpy as np
import cv2
import random

# Add src path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/semantic_training')))

from env import Env

def run_demo():
    output_dir = "output/demo_vis_generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Initializing environment... Output: {os.path.abspath(output_dir)}")
    env = Env(map_index=0, plot=True, data_dir="generated_data")
    
    # Save ground truth
    env.save_ground_truth(output_dir)
    
    # Initial Rotation
    orientations = np.linspace(0, 2*np.pi, 18)[:-1]
    frames = []
    step_idx = 0
    
    def capture_frame(step_label):
        # 1. Main View (RGB)
        img_main = env.plot_env(global_step=0, path=None, step_idx=step_idx, return_img=True)
        
        # 2. Entropy View (RGB)
        # Invert normalization: High Entropy (1.0) -> 0.0, Low Entropy (0.0) -> 1.0
        # This makes Known areas (Low Ent) -> Bright/Hot, Unknown (High Ent) -> Dark/Cold
        entropy_inv = 1.0 - env.entropy_map
        entropy_norm = (entropy_inv * 255).astype(np.uint8)
        
        # Use HOT colormap: Black (0.0/High Ent) -> Red -> Yellow -> White (1.0/Low Ent)
        # So Unknown is Black, Known is Bright.
        img_entropy = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_HOT)
        img_entropy = cv2.cvtColor(img_entropy, cv2.COLOR_BGR2RGB)
        
        # Add Legend
        # Draw a gradient bar at the bottom
        h, w = img_entropy.shape[:2]
        bar_height = 20
        bar_y = h - 30
        
        # Create gradient
        gradient = np.linspace(0, 255, w).astype(np.uint8)
        gradient_img = np.tile(gradient, (bar_height, 1))
        gradient_color = cv2.applyColorMap(gradient_img, cv2.COLORMAP_HOT)
        gradient_color = cv2.cvtColor(gradient_color, cv2.COLOR_BGR2RGB)
        
        # Overlay gradient
        img_entropy[bar_y:bar_y+bar_height, :] = gradient_color
        
        # Labels
        cv2.putText(img_entropy, "High Entropy (Unknown)", (5, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_entropy, "Low Entropy (Known)", (w - 150, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img_entropy, "Entropy Map", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 3. Semantic View (RGB)
        img_sem = np.zeros_like(img_main)
        # Background: Dark Gray for explored free space, Black for obstacles/unexplored
        img_sem[:] = [20, 20, 20] 
        
        visible_mask = (env.robot_belief != 127)
        # Draw Free Space lightly
        img_sem[(env.robot_belief == 255)] = [50, 50, 50]
        
        # Draw Obstacles
        img_sem[(env.robot_belief == 1)] = [0, 0, 0]
        
        # Draw Semantics
        # We want to distinguish "Found" (Confirmed) vs "Observed but not confirmed" (if any)
        # But in this env, we only track found_semantics. The semantic map has ground truth IDs.
        # We should only draw semantics that are in the "visible" area (robot_belief != 127)
        
        # Get all IDs in the map
        all_ids = np.unique(env.semantic_map)
        
        np.random.seed(42)
        colors = {uid: np.random.randint(50, 255, 3).tolist() for uid in all_ids if uid > 0}
        
        for uid, color in colors.items():
            # Only draw if visible (explored)
            mask = (env.semantic_map == uid) & visible_mask
            if np.any(mask):
                if uid in env.found_semantics:
                    # Confirmed: Bright Color
                    img_sem[mask] = color
                    
                    # Draw Bounding Box or Text
                    y, x = np.where(mask)
                    if len(y) > 0:
                        cy, cx = int(np.mean(y)), int(np.mean(x))
                        # Use contrasting text color
                        cv2.putText(img_sem, "CONFIRMED", (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                else:
                    # Visible but not confirmed (e.g. low probability? or just entered view)
                    # Make it dim
                    dim_color = [c // 3 for c in color]
                    img_sem[mask] = dim_color
            
        cv2.putText(img_sem, f"Semantics: {len(env.found_semantics)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Step Count
        info_bar = np.zeros((30, img_main.shape[1] * 3, 3), dtype=np.uint8)
        cv2.putText(info_bar, f"Step: {step_idx} | {step_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        combined = np.hstack((img_main, img_entropy, img_sem))
        final_frame = np.vstack((combined, info_bar))
        
        frames.append(final_frame)
        cv2.imwrite(f"{output_dir}/frame_{step_idx:04d}.png", cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

    print("Phase 1: Rotation...")
    for angle in orientations:
        env.step(env.robot_position, next_orientation=angle)
        env.frontiers = env.find_frontier()
        env.update_graph()
        capture_frame(f"Rotate")
        step_idx += 1
        
    print("Phase 2: Graph-based Exploration...")
    
    env.frontiers = env.find_frontier()
    env.update_graph()
    
    max_steps = 200
    current_step = 0
    
    while current_step < max_steps:
        # 1. Find current node
        current_idx = env.find_index_from_coords(env.robot_position)
        if current_idx is None:
            # Try to snap to nearest node if close enough
            dists = np.linalg.norm(env.node_coords - env.robot_position, axis=1)
            if np.min(dists) < 5.0:
                current_idx = np.argmin(dists)
            else:
                print("Lost! Not near any node.")
                break
            
        str_idx = str(current_idx)
        if not hasattr(env, 'graph') or str_idx not in env.graph:
            env.update_graph()
            if str_idx not in env.graph:
                 print("Node has no edges. Stopping.")
                 break
            
        # 2. Get neighbors
        neighbors = env.graph[str_idx]
        if not neighbors:
            print("Dead end.")
            # Backtrack? For demo, just stop.
            break
            
        # 3. Choose a neighbor
        # Heuristic: Prefer Unvisited Nodes, then align with orientation
        # Since we don't have global visited set for nodes in this script easily (env.node_utility?), 
        # let's just pick random weighted by alignment to explore.
        
        valid_neighbors = []
        vx = np.cos(env.robot_orientation)
        vy = np.sin(env.robot_orientation)
        
        for n_idx in neighbors:
            n_idx = int(n_idx)
            if n_idx >= len(env.node_coords): continue
            n_pos = env.node_coords[n_idx]
            
            dx = n_pos[0] - env.robot_position[0]
            dy = n_pos[1] - env.robot_position[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-3: continue
            
            # Direction score
            align_score = (dx * vx + dy * vy) / dist
            
            # Random score to encourage exploration
            rand_score = random.uniform(0, 1.0)
            
            final_score = align_score + rand_score
            valid_neighbors.append((n_idx, n_pos, final_score))
            
        if not valid_neighbors:
            break
            
        # Pick best
        valid_neighbors.sort(key=lambda x: x[2], reverse=True)
        target_idx, target_pos, score = valid_neighbors[0]
        
        # 4. Move to target
        start_pos = env.robot_position.copy()
        end_pos = target_pos
        dist = np.linalg.norm(end_pos - start_pos)
        
        # Move in steps
        move_step_size = 5
        n_interp = int(dist / move_step_size) + 1
        
        target_angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        
        for s in range(n_interp):
            if current_step >= max_steps: break
            
            t = min(1.0, (s + 1) / n_interp)
            interp_pos = start_pos * (1 - t) + end_pos * t
            
            env.step(interp_pos, next_orientation=target_angle)
            env.frontiers = env.find_frontier()
            env.update_graph()
            
            capture_frame(f"Move to {target_idx}")
            step_idx += 1
            current_step += 1
            
        # Ensure final position
        if current_step < max_steps:
            env.step(end_pos, next_orientation=target_angle)
            env.frontiers = env.find_frontier()
            env.update_graph()
            capture_frame(f"Arrived {target_idx}")
            step_idx += 1
            current_step += 1
            
    # Save GIF
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        gif_path = os.path.join(output_dir, "simulation_demo.gif")
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=100, loop=0) # Faster 100ms
        print(f"GIF saved to: {os.path.abspath(gif_path)}")
    except ImportError:
        print("PIL not found. GIF not saved.")

if __name__ == "__main__":
    run_demo()
