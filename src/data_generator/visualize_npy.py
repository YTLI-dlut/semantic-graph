import argparse
import os
import json
import numpy as np
import cv2

def hex_to_rgb(hex_str):
    """ 'DF6159' -> (R, G, B) """
    if not hex_str: return (127, 127, 127)
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def visualize(dataset_dir):
    geo_path = os.path.join(dataset_dir, "grid_geometric.npy")
    sem_path = os.path.join(dataset_dir, "grid_semantic.npy")
    json_path = os.path.join(dataset_dir, "instances.json")
    
    if not os.path.exists(geo_path) or not os.path.exists(sem_path):
        print(f"Error: NPY files not found in {dataset_dir}")
        return

    print(f"Loading {geo_path}...")
    grid_geo = np.load(geo_path)
    print(f"Loading {sem_path}...")
    grid_sem = np.load(sem_path)
    
    # 1. Visualize Geometric Map
    # Values: 0 (Obstacle), 255 (Free), 127 (Unknown)
    # We can just save it as is because it's uint8 grayscale compatible.
    vis_geo = grid_geo.astype(np.uint8)
    # create a colored version for better contrast if needed
    vis_geo_color = cv2.cvtColor(vis_geo, cv2.COLOR_GRAY2BGR)
    # Make unknown (127) gray, Free (255) white, Obstacle (0) Black.
    # It is already like that.
    
    out_geo_path = os.path.join(dataset_dir, "check_vis_geometric.png")
    cv2.imwrite(out_geo_path, vis_geo)
    print(f"Saved geometric visualization to {out_geo_path}")

    # 2. Visualize Semantic Map
    instances = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            instances = json.load(f)
            
    H, W = grid_sem.shape
    vis_sem = np.full((H, W, 3), 200, dtype=np.uint8) # Default gray background
    
    # Unique IDs in the map
    unique_ids = np.unique(grid_sem)
    
    print(f"Found {len(unique_ids)} unique semantic IDs in map.")
    
    for uid in unique_ids:
        if uid == -1: continue # Background
        if uid == 0: continue
        
        # Get color
        color_bgr = (0, 0, 0)
        str_uid = str(uid)
        
        if str_uid in instances:
            # Use color from json
            hex_color = instances[str_uid].get("color_hex", "")
            r, g, b = hex_to_rgb(hex_color)
            color_bgr = (b, g, r) # OpenCV is BGR
        else:
            # Random color for unknown IDs
            np.random.seed(int(uid))
            color_bgr = np.random.randint(0, 255, 3).tolist()
            
        # Draw
        mask = (grid_sem == uid)
        vis_sem[mask] = color_bgr

    out_sem_path = os.path.join(dataset_dir, "check_vis_semantic.png")
    cv2.imwrite(out_sem_path, vis_sem)
    print(f"Saved semantic visualization to {out_sem_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Path to dataset directory. Defaults to latest.")
    args = parser.parse_args()
    
    target_dir = args.dir
    if target_dir is None:
        base_dir = "/home/iiau/createGraph_ws/output/manual_maps"
        if os.path.exists(base_dir):
            all_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if all_dirs:
                # Find latest by modification time
                target_dir = max(all_dirs, key=os.path.getmtime)
                print(f"Auto-selected latest directory: {target_dir}")
    
    if target_dir and os.path.exists(target_dir):
        # 0. Print Metadata
        meta_path = os.path.join(target_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print("\n=== Metadata ===")
            print(json.dumps(meta, indent=2))
            print("================\n")
            
        visualize(target_dir)
    else:
        print(f"Directory not found: {target_dir}")
