
import os
import argparse
import json
import numpy as np
import cv2

class InteractiveViewer:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.load_data()
        
        # Windows
        self.window_name = "Interactive Semantic Viewer"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        # State
        self.cur_mouse_pos = (0, 0)
        self.hovered_info = None
        self.hovered_mask = None
        self.running = True

    def load_data(self):
        print(f"Loading dataset from: {self.dataset_dir}")
        
        # Paths
        geo_path = os.path.join(self.dataset_dir, "grid_geometric.npy")
        sem_path = os.path.join(self.dataset_dir, "grid_semantic.npy")
        inst_path = os.path.join(self.dataset_dir, "instances.json")
        
        if not os.path.exists(geo_path) or not os.path.exists(sem_path):
            raise FileNotFoundError("NPY files not found.")
            
        self.grid_geo = np.load(geo_path) # (H, W) uint8
        self.grid_sem = np.load(sem_path) # (H, W) int32
        
        self.instances = {}
        if os.path.exists(inst_path):
            with open(inst_path, 'r') as f:
                self.instances = json.load(f)
                
        # Pre-render base semantic map
        self.H, self.W = self.grid_sem.shape
        self.vis_base = np.full((self.H, self.W, 3), 40, dtype=np.uint8) # Dark gray bg
        
        # Colorize
        unique_ids = np.unique(self.grid_sem)
        print(f"Found {len(unique_ids)} unique instances.")
        
        for uid in unique_ids:
            if uid <= 0: continue
            
            # Get color
            color = (0, 0, 0)
            str_uid = str(uid)
            if str_uid in self.instances:
                hex_c = self.instances[str_uid].get("color_hex", "")
                if hex_c:
                    h = hex_c.lstrip('#')
                    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                    color = (rgb[2], rgb[1], rgb[0]) # BGR
            else:
                np.random.seed(int(uid))
                color = np.random.randint(50, 255, 3).tolist()
                
            mask = (self.grid_sem == uid)
            self.vis_base[mask] = color
            
        # Draw geometric obstacles as overlay (Black) to define boundaries better
        # Geometric: 0=Obstacle, 255=Free, 127=Unknown
        # We only care about Obstacle lines sometimes, or we can just ignore geometric
        # Let's blend geometric obstacles on top lightly
        obs_mask = (self.grid_geo == 0)
        self.vis_base[obs_mask] = (self.vis_base[obs_mask] * 0.5).astype(np.uint8)


    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cur_mouse_pos = (x, y)
            self.update_info(x, y)

    def update_info(self, x, y):
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            self.hovered_info = None
            self.hovered_mask = None
            return
            
        inst_id = self.grid_sem[y, x]
        
        if inst_id > 0:
            str_uid = str(inst_id)
            info = self.instances.get(str_uid, {"category_name": "unknown"})
            cat = info.get("category_name", "unknown")
            rid = info.get("room_id", -1)
            
            self.hovered_info = f"ID: {inst_id} | {cat} | Room: {rid}"
            # Create highlight mask
            self.hovered_mask = (self.grid_sem == inst_id)
        else:
            self.hovered_info = None
            self.hovered_mask = None

    def run(self):
        print("Starting viewer... Press 'ESC' or 'q' to quit.")
        
        while self.running:
            # Start with base image
            display_img = self.vis_base.copy()
            
            # Draw highlight
            if self.hovered_mask is not None:
                # Add white highlight to the selected object
                # Create a white overlay
                overlay = np.zeros_like(display_img)
                overlay[self.hovered_mask] = (255, 255, 255)
                
                # Blend
                cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0, display_img)
                
                # Draw contour boundaries
                contours, _ = cv2.findContours(self.hovered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_img, contours, -1, (255, 255, 255), 2)

            # Draw UI Text
            if self.hovered_info:
                # Top bar background
                cv2.rectangle(display_img, (0, 0), (self.W, 40), (0, 0, 0), -1)
                cv2.putText(display_img, self.hovered_info, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.rectangle(display_img, (0, 0), (self.W, 40), (0, 0, 0), -1)
                cv2.putText(display_img, "Hover over objects...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
                            
            # Show
            cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(20)
            if key == 27 or key == ord('q'):
                self.running = False
                
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Dataset directory")
    args = parser.parse_args()
    
    target_dir = args.dir
    if target_dir is None:
        base_dir = "/home/iiau/createGraph_ws/output/manual_maps"
        if os.path.exists(base_dir):
            all_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if all_dirs:
                target_dir = max(all_dirs, key=os.path.getmtime)
    
    if target_dir and os.path.exists(target_dir):
        viewer = InteractiveViewer(target_dir)
        viewer.run()
    else:
        print("No dataset found.")

if __name__ == "__main__":
    main()
