
import os
import numpy as np
import time
import math
import cv2
from PIL import Image, ImageDraw, ImageFont

from sensor import *
from graph_generator import *
from node import *
from parameter import *
from data_loader import DataLoader

if not train_mode:
    from test_parameter import *

class Env():
    def __init__(self, map_index, k_size=20, plot=False, test=False, data_dir="generated_data"):
        self.test = test
        self.k_size = k_size
        self.plot = plot
        
        # DataLoader
        self.data_loader = DataLoader(data_dir=data_dir)
        if not self.data_loader.map_list:
            raise RuntimeError(f"No maps found in {data_dir}!")
            
        self.ground_truth_data = self.data_loader.get_random_map()
        self.ground_truth = self.ground_truth_data['geometric']
        self.semantic_map = self.ground_truth_data['semantic']
        self.map_size = self.ground_truth.shape 
        
        # Robot State (Single Agent)
        self.robot_position = self._find_valid_start_position()
        self.robot_orientation = 0.0 # Radians
        self.fov = np.radians(120)   # 120 degrees
        self.robot_belief = np.full(self.map_size, 127, dtype=np.uint8)
        
        # Entropy Map
        self.entropy_map = np.full(self.map_size, 1.0, dtype=np.float32) # Max entropy normalized to 1.0
        
        # Graph Generator
        # sensor_range: 3.5m / 0.05 resolution = 70 pixels
        self.sensor_range = 70 
        self.graph_generator = Graph_generator(self.map_size, self.k_size, self.sensor_range, plot)
        
        self.old_robot_belief = self.robot_belief.copy()
        self.explored_rate = 0
        self.done = False
        
        # Initialize Fonts
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        ]
        self.font = None
        for fp in font_paths:
            try:
                self.font = ImageFont.truetype(fp, 18)
                self.title_font = ImageFont.truetype(fp, 22)
                self.legend_font = ImageFont.truetype(fp, 14)
                break
            except:
                continue
        
        if self.font is None:
            self.font = ImageFont.load_default()
            self.title_font = self.font
            self.legend_font = self.font
        
        # Initial Sense
        self.update_robot_belief(self.robot_position, self.sensor_range, self.robot_belief, self.ground_truth)
        
        # History
        self.robot_path = [self.robot_position]
        self.travel_dist = 0
        self.steps = 0
        self.found_semantics = set()
        self.seen_semantics = set()
        self.visited_nodes = set()
        
        # Initial Graph
        self.frontiers = self.find_frontier() # Init frontiers
        self.old_frontiers = set(tuple(f) for f in self.frontiers) # Init old_frontiers set
        self.update_graph()

    def update_graph(self):
        self.frontiers = self.find_frontier()
        # Note: old_frontiers is updated in calculate_reward
        self.node_coords, self.graph, self.node_utility, self.guidepost, self.node_k_flags = \
            self.graph_generator.generate_graph([self.robot_position], self.robot_belief, self.frontiers, k_traj=[self.robot_path])

    def _find_valid_start_position(self):
        for _ in range(1000):
            r = np.random.randint(0, self.map_size[0])
            c = np.random.randint(0, self.map_size[1])
            if self.ground_truth[r, c] == 255: 
                return np.array([c, r], dtype=float) # Return (x, y)
        return np.array([self.map_size[1]//2, self.map_size[0]//2], dtype=float)

    def step(self, next_position, next_orientation=None):
        # Move
        self.robot_position = next_position
        if next_orientation is not None:
            self.robot_orientation = next_orientation
        else:
            # Default behavior: orient towards movement direction if moved significantly
            # Or keep previous orientation. Let's keep previous for now to allow strafing if needed,
            # but ideally the agent controls orientation.
            pass

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
        self.robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth, 
                                      robot_orientation=self.robot_orientation, fov=self.fov)
        
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
        
        # 1. Distance check
        dist_sq = (X - cx)**2 + (Y - cy)**2
        dist = np.sqrt(dist_sq)
        mask_circle = dist_sq <= r**2
        
        # 2. Angle check (FOV)
        angles = np.arctan2(Y - cy, X - cx)
        angle_diff = angles - self.robot_orientation
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        mask_fov = np.abs(angle_diff) <= (self.fov / 2)
        
        # 3. Visibility check
        mask_observed = (local_belief != 127) & mask_circle & mask_fov
        
        visible_ids = np.unique(local_sem[mask_observed])
        new_count = 0
        
        # Track seen semantics (Discovery)
        self.new_seen_count = 0
        for uid in visible_ids:
            if uid > 0 and uid not in self.seen_semantics:
                self.seen_semantics.add(uid)
                self.new_seen_count += 1
        
        # Calculate Probabilities
        p_dist = np.maximum(0, 1 - dist / r)
        p_angle = np.cos(np.abs(angle_diff) / (self.fov / 2) * (np.pi / 2))
        p_angle[~mask_fov] = 0
        p_recog = p_dist * p_angle
        
        # --- Single Map Mask-based Entropy Update ---
        
        # 1. Generate Masks
        # mask_free: Free space (255)
        mask_free = (self.robot_belief == 255)
        
        # mask_high_entropy: Entropy > 0
        mask_high_entropy = (self.entropy_map > 0)
        
        # mask_interest: Near Unconfirmed Objects or Frontiers
        mask_interest = np.zeros(self.map_size, dtype=bool)
        
        unconfirmed = self.get_unconfirmed_centers()
        # Frontiers as list of coords
        frontiers_list = list(self.frontiers) if len(self.frontiers) > 0 else []
        
        # Draw circles for interest regions (Efficient approximation of sensor range check)
        # Using cv2 for speed on bool mask? No, cv2 needs uint8.
        temp_mask = np.zeros(self.map_size, dtype=np.uint8)
        
        # Targets: Unconfirmed + Frontiers
        targets = []
        if len(unconfirmed) > 0:
            targets.extend(unconfirmed)
        if len(frontiers_list) > 0:
            targets.extend(frontiers_list)
            
        for target in targets:
            # Draw filled circle
            # ENTROPY_BOOST_RADIUS from parameter.py (e.g., 30)
            # Use max(sensor_range, boost_radius) or just sensor_range?
            # User said "sensor range of unconfirmed objects". 
            # self.sensor_range is 70.
            cv2.circle(temp_mask, (int(target[0]), int(target[1])), self.sensor_range, 1, -1)
            
        mask_interest = (temp_mask == 1)
        
        # update_mask = Free & (HighEntropy | Interest)
        update_mask = mask_free & (mask_high_entropy | mask_interest)
        
        # 2. Apply Updates
        
        # A. Observation Decay (Local)
        # Apply only where observed AND in update_mask
        # Note: mask_observed is local window coordinates. update_mask is global.
        # Need to slice update_mask to local window
        local_update_mask = update_mask[y_min:y_max, x_min:x_max]
        local_entropy = self.entropy_map[y_min:y_max, x_min:x_max]
        
        # Intersection of Observed and Update Candidate
        mask_decay = mask_observed & local_update_mask
        
        if np.any(mask_decay):
            learning_rate = 0.5
            entropy_reduction = p_recog * learning_rate
            local_entropy[mask_decay] *= (1 - entropy_reduction[mask_decay])
            
        # Write back decayed entropy to map (needed before Boost step if we want to combine, 
        # but Boost is global. Let's write back first)
        self.entropy_map[y_min:y_max, x_min:x_max] = local_entropy
        
        # B. Interest Boost (Global)
        # Apply to all pixels in mask_interest & update_mask (Global coords)
        # For efficiency, we can just iterate the targets again and update their ROIs directly,
        # masking with mask_free.
        
        # However, we must ensure we don't overwrite the Decay we just did if it overlaps?
        # User logic: "Interest" implies high entropy. 
        # If Robot is observing it, Decay fights Boost.
        # Order: Decay then Boost? Or Boost then Decay?
        # If Boost sets to MAX, and Decay reduces it. 
        # If we do Boost AFTER Decay, the area under robot will stay High if it's an interest zone.
        # This effectively means "Agent cannot reduce entropy of interest zone until confirmed".
        # This seems correct for "Attraction".
        
        # Boost Value
        boost_val = ENTROPY_UNCONFIRMED_BOOST # e.g. 2.0. Map is 0.0-1.0 usually? 
        # Initial entropy is 1.0. Boost 2.0 is very high.
        # If we just set it: entropy = max(entropy, boost_val)
        
        # Optimization: Only iterate targets, don't update full map mask
        for target in targets:
            cx, cy = int(target[0]), int(target[1])
            r_boost = self.sensor_range
            
            x1, x2 = max(0, cx - r_boost), min(self.map_size[1], cx + r_boost)
            y1, y2 = max(0, cy - r_boost), min(self.map_size[0], cy + r_boost)
            
            # ROI slices
            roi_entropy = self.entropy_map[y1:y2, x1:x2]
            roi_free = mask_free[y1:y2, x1:x2]
            
            # Create circular mask for this ROI
            h_roi, w_roi = roi_entropy.shape
            Y_roi, X_roi = np.ogrid[:h_roi, :w_roi]
            dist_sq_roi = (X_roi - (cx - x1))**2 + (Y_roi - (cy - y1))**2
            mask_roi_circle = dist_sq_roi <= r_boost**2
            
            # Update Mask for this ROI: Free & Circle
            mask_roi_update = roi_free & mask_roi_circle
            
            # Apply Boost
            # Linear decay boost? "Calculate their entropy".
            # Old logic: boost += 2.0 * (1 - dist/r).
            # Here we set value directly? 
            # Let's use: entropy = max(entropy, 1.0 + boost_val * (1 - dist/r))
            # Or just simplified max(entropy, boost_val)?
            # User said "Calculate...". Let's use distance based boost for better gradient.
            
            if np.any(mask_roi_update):
                dist_roi = np.sqrt(dist_sq_roi[mask_roi_update])
                # Boost formula: 1.0 (Base Max) + Boost * (1 - d/r)
                # This ensures it's > 1.0 near center.
                boosted_values = 1.0 + ENTROPY_UNCONFIRMED_BOOST * (1 - dist_roi / r_boost)
                
                # Apply max
                roi_entropy[mask_roi_update] = np.maximum(roi_entropy[mask_roi_update], boosted_values)
                self.entropy_map[y1:y2, x1:x2] = roi_entropy

        
        for uid in visible_ids:
            if uid > 0 and uid not in self.found_semantics:
                obj_mask = (local_sem == uid) & mask_observed
                if np.any(obj_mask):
                    avg_prob = np.mean(p_recog[obj_mask])
                    if np.random.random() < avg_prob:
                        self.found_semantics.add(uid)
                        new_count += 1
        return new_count

    def calculate_reward(self, move_dist):
        # 1. Frontier Elimination Reward
        current_frontiers_set = set(tuple(f) for f in self.frontiers)
        intersection = self.old_frontiers & current_frontiers_set
        eliminated_count = len(self.old_frontiers) - len(intersection)
        reward_explore = max(0, eliminated_count) * REWARD_EXPLORE_CELL
        
        self.old_frontiers = current_frontiers_set # Update for next step
        
        # Entropy Reward (Removed as requested)
        # current_total_entropy = np.sum(self.entropy_map)
        # if not hasattr(self, 'last_total_entropy'):
        #      self.last_total_entropy = self.map_size[0] * self.map_size[1]
             

        reward = reward_explore
        reward -= move_dist * 0.05 
        
        # Discovery Reward
        reward += self.new_seen_count * REWARD_NEW_SEEN

        # Node Coverage Reward
        # Track visited positions (quantized to 10 pixels ~ 0.5m)
        q_pos = (int(self.robot_position[0] // 10), int(self.robot_position[1] // 10))
        coverage_reward = 0.0
        if q_pos not in self.visited_nodes:
            self.visited_nodes.add(q_pos)
            coverage_reward = 0.2
            reward += coverage_reward
            
        # Repeat Penalty (Coverage Mechanism against staying still)
        # History queue: self.pose_history = deque(maxlen=10)
        # Format: (x, y, ori_idx)
        repeat_penalty = 0.0
        
        # Quantize current state for history matching
        # Position: 10 pixels resolution, Orientation: 0.1 rad resolution
        curr_state = (int(self.robot_position[0]), int(self.robot_position[1]), round(self.robot_orientation, 2))
        
        if not hasattr(self, 'pose_history'):
            from collections import deque
            self.pose_history = deque(maxlen=10)
            
        # Check match
        if curr_state in self.pose_history:
            repeat_penalty = REWARD_REPEAT_PENALTY
            reward += repeat_penalty
        
        # Update history
        self.pose_history.append(curr_state)

        # Store detailed rewards for metrics
        self.last_rewards = {
            "explore": reward_explore,
            "discovery": self.new_seen_count * REWARD_NEW_SEEN,
            "coverage": coverage_reward,
            "repeat": repeat_penalty
        }
        
        self.old_robot_belief = self.robot_belief.copy()
        return reward

    def get_unconfirmed_centers(self):
        # Mask: Observed & Semantic & NOT Confirmed
        mask_observed = (self.robot_belief != 127)
        mask_semantic = (self.semantic_map > 0)
        mask_confirmed = np.isin(self.semantic_map, list(self.found_semantics))
        
        target_mask = mask_observed & mask_semantic & (~mask_confirmed)
        target_mask = target_mask.astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(target_mask, connectivity=8)
        
        valid_centroids = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 5:
                 valid_centroids.append(centroids[i])
        return valid_centroids

    def get_node_features(self, node_coords):
        # 1. Entropy Feature
        # Optimization: Directly sample from the updated global entropy_map
        # The entropy_map now includes both exploration entropy and unconfirmed object boosts
        entropy_feats = []
        r = 15 
        h, w = self.map_size
        
        for coord in node_coords:
             cx, cy = int(coord[0]), int(coord[1])
             x1, x2 = max(0, cx-r), min(w, cx+r)
             y1, y2 = max(0, cy-r), min(h, cy+r)
             local_ent = self.entropy_map[y1:y2, x1:x2]
             
             avg_entropy = 0.0
             if local_ent.size > 0:
                 avg_entropy = np.mean(local_ent)
             
             entropy_feats.append(avg_entropy)
             
        entropy_feats = np.array(entropy_feats).reshape(-1, 1)
        
        # 2. Unconfirmed Object Vector
        unconfirmed_centers = self.get_unconfirmed_centers()
        vector_feats = []
        
        norm_scale = max(h, w)
        
        for coord in node_coords:
             if not unconfirmed_centers:
                 vector_feats.append([0.0, 0.0])
                 continue
                 
             dists = np.linalg.norm(unconfirmed_centers - coord, axis=1)
             nearest_idx = np.argmin(dists)
             nearest_center = unconfirmed_centers[nearest_idx]
             
             diff = nearest_center - coord
             diff /= norm_scale
             vector_feats.append(diff)
             
        vector_feats = np.array(vector_feats)
        return entropy_feats, vector_feats

    def check_done(self):
        # Done if no frontiers AND all seen semantics are confirmed
        all_confirmed = (len(self.seen_semantics) > 0) and (self.seen_semantics == self.found_semantics)
        # If no semantics seen yet, rely on frontiers. If seen, must confirm all.
        # But logically, if frontiers=0, we explored everything reachable.
        # So: Done = (Frontiers == 0) AND (All Seen Confirmed)
        # However, if we can't reach the object (no frontiers), we might be stuck?
        # Assuming reachable:
        if len(self.frontiers) == 0:
            if len(self.seen_semantics) == 0:
                self.done = True
            elif self.seen_semantics == self.found_semantics:
                self.done = True
        return self.done
        
    def find_frontier(self):
        return get_frontier_in_map(self.robot_belief)

    def plot_env(self, global_step, path, step_idx):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
    def plot_env(self, global_step, path, step_idx, info_text="", return_img=False):
        # 1. Base Map (RGB)
        h, w = self.map_size
        base_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 255 (Free) -> White, 1 (Obstacle) -> Black, 127 (Unknown) -> Gray
        # Note: BGR format for OpenCV
        base_img[self.robot_belief == 255] = [255, 255, 255]
        base_img[self.robot_belief == 1] = [0, 0, 0]
        base_img[self.robot_belief == 127] = [127, 127, 127]
        
        # --- Panel 1: Simulation View (Robot, Path, Frontiers, Nodes) ---
        sim_img = base_img.copy()
        
        # Path
        if len(self.robot_path) > 1:
            pts = np.array(self.robot_path, np.int32).reshape((-1, 1, 2))
            cv2.polylines(sim_img, [pts], False, (255, 0, 0), 1) # Blue in BGR
        
        # Robot
        rx, ry = int(self.robot_position[0]), int(self.robot_position[1])
        cv2.circle(sim_img, (rx, ry), 3, (255, 0, 0), -1) # Blue robot (BGR)
        
        # FOV
        start_angle_deg = np.degrees(self.robot_orientation - self.fov/2)
        end_angle_deg = np.degrees(self.robot_orientation + self.fov/2)
        
        overlay = sim_img.copy()
        # Yellow in BGR is (0, 255, 255)
        cv2.ellipse(overlay, (rx, ry), (self.sensor_range, self.sensor_range), 
                   0, start_angle_deg, end_angle_deg, (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, sim_img, 0.7, 0, sim_img)
        
        # Frontiers
        for f in self.frontiers:
            cv2.circle(sim_img, (int(f[0]), int(f[1])), 1, (0, 0, 255), -1) # Red in BGR
            
        # Nodes
        if self.plot and self.node_coords is not None:
             for node in self.node_coords:
                 cv2.circle(sim_img, (int(node[0]), int(node[1])), 2, (0, 255, 0), -1) # Green
             
             # Edges
             current_idx = self.find_index_from_coords(self.robot_position)
             if current_idx is not None and current_idx < len(self.node_coords):
                 str_idx = str(current_idx)
                 if hasattr(self, 'graph') and str_idx in self.graph:
                     neighbors = self.graph[str_idx]
                     for n_idx in neighbors:
                         n_idx = int(n_idx)
                         if n_idx < len(self.node_coords):
                             n_pos = self.node_coords[n_idx]
                             cv2.line(sim_img, (rx, ry), (int(n_pos[0]), int(n_pos[1])), (255, 255, 0), 1) # Cyan

        # --- Panel 2: Entropy View ---
        # Normalize
        e_min = self.entropy_map.min()
        e_max = self.entropy_map.max()
        entropy_norm = (self.entropy_map - e_min) / (e_max - e_min + 1e-6)
        entropy_norm = (entropy_norm * 255).astype(np.uint8)
        entropy_img = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_JET)
        
        # Mask Unknown Areas (Background -> Black)
        # Assuming robot_belief == 127 is Unknown
        entropy_img[self.robot_belief == 127] = [0, 0, 0]
        
        # Add Legend (Gradient Bar)
        # Size: 30% of width, height 12px
        leg_w = int(w * 0.4)
        leg_h = 12
        leg_pad = 10
        
        # Gradient
        grad = np.linspace(0, 255, leg_w).astype(np.uint8)
        grad = np.tile(grad, (leg_h, 1))
        grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
        
        # Position: Bottom Right
        x_start = w - leg_w - leg_pad
        y_start = h - leg_h - 25 # Leave space for text below
        
        # Overlay
        # Check bounds
        if y_start > 0 and x_start > 0:
            entropy_img[y_start:y_start+leg_h, x_start:x_start+leg_w] = grad_color
            
        # Robot Marker
        cv2.circle(entropy_img, (rx, ry), 3, (255, 255, 255), -1)

        # --- Panel 3: Semantic View ---
        sem_img = np.zeros_like(base_img)
        # Background: Dimmed Ground Truth
        sem_img[self.ground_truth == 255] = [220, 220, 220]
        sem_img[self.ground_truth == 1] = [50, 50, 50]
        sem_img[self.ground_truth == 127] = [127, 127, 127]

        # Objects
        uids = np.unique(self.semantic_map)
        for uid in uids:
            if uid == 0: continue
            mask = (self.semantic_map == uid)
            # Find centroid
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                color = (100, 100, 100) # Unseen - Dark Gray
                radius = 3
                if uid in self.found_semantics:
                    color = (0, 255, 0) # Confirmed - Green
                    radius = 5
                elif uid in self.seen_semantics:
                    color = (0, 0, 255) # Seen - Red
                    radius = 4
                
                cv2.circle(sem_img, (cX, cY), radius, color, -1)
        
        # Robot
        cv2.circle(sem_img, (rx, ry), 3, (255, 0, 0), -1)

        # --- Combine ---
        combined_img = np.hstack([sim_img, entropy_img, sem_img])
        
        # --- Add Chinese Titles and Info ---
        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Titles
        draw.text((10, 10), "仿真视角", font=self.title_font, fill=(255, 0, 0))
        draw.text((w + 10, 10), "熵值热力图", font=self.title_font, fill=(255, 0, 0))
        draw.text((2*w + 10, 10), "语义地图(绿:确认,红:发现)", font=self.title_font, fill=(255, 0, 0))
        
        # Entropy Legend Values
        # Re-calculate legend dimensions to match OpenCV drawing
        leg_w = int(w * 0.4)
        leg_h = 12
        leg_pad = 10
        x_offset = w # Middle panel offset
        
        l_x_start = x_offset + (w - leg_w - leg_pad)
        l_y_start = h - leg_h - 25
        l_y_text = l_y_start + leg_h + 2
        
        # Draw Min/Max values (White text for contrast on black background)
        if 'e_min' in locals() and 'e_max' in locals():
            draw.text((l_x_start, l_y_text), f"{e_min:.2f}", font=self.legend_font, fill=(255, 255, 255))
            max_str = f"{e_max:.2f}"
            # Estimate width
            try:
                max_w = draw.textlength(max_str, font=self.legend_font)
            except:
                max_w = self.legend_font.getsize(max_str)[0]
            draw.text((l_x_start + leg_w - max_w, l_y_text), max_str, font=self.legend_font, fill=(255, 255, 255))

        # Info Text
        if info_text:
            y0, dy = 50, 25
            for i, line in enumerate(info_text.split('\n')):
                y = y0 + i*dy
                draw.text((10, y), line, font=self.font, fill=(255, 0, 0))

        # Convert back to OpenCV
        combined_img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        if return_img:
            return cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(f'{path}/step_{step_idx:04d}.png', combined_img)

    def save_ground_truth(self, path):
         img = np.zeros((self.map_size[0], self.map_size[1], 3), dtype=np.uint8)
         img[self.ground_truth == 255] = [255, 255, 255]
         img[self.ground_truth == 1] = [0, 0, 0]
         img[self.ground_truth == 127] = [127, 127, 127]
         
         if not os.path.exists(path):
            os.makedirs(path)
         cv2.imwrite(f'{path}/ground_truth.png', img)

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
