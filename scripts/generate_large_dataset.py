import os
import json
import numpy as np
import cv2
import random
import time
from datetime import datetime

class MapGenerator:
    def __init__(self, output_base_dir, num_scenes=500, map_size=(500, 500), resolution=0.05,
                 obj_size_range=(10, 20), min_obj_spacing=20, wall_padding=15, room_count_range=(3, 8)):
        self.output_base_dir = output_base_dir
        self.num_scenes = num_scenes
        self.map_size = map_size
        self.resolution = resolution
        # Configuration for object generation
        self.obj_size_range = obj_size_range   # (min, max) size in pixels
        self.min_obj_spacing = min_obj_spacing # Minimum distance between objects
        self.wall_padding = wall_padding       # Minimum distance from walls/corners
        self.room_count_range = room_count_range # (min, max) number of rooms

        
        self.categories = [
            {"name": "table", "color": "#8B4513"},
            {"name": "chair", "color": "#CD853F"},
            {"name": "sofa", "color": "#4682B4"},
            {"name": "bed", "color": "#483D8B"},
            {"name": "toilet", "color": "#F0F8FF"},
            {"name": "shelf", "color": "#D2691E"},
            {"name": "cabinet", "color": "#A0522D"},
            {"name": "plant", "color": "#228B22"}
        ]

    def hex_to_rgb(self, hex_str):
        h = hex_str.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def generate_single_scene(self, scene_idx):
        # Initialize maps
        # Geometric: 0=Obstacle (Wall), 255=Free
        grid_geo = np.zeros(self.map_size, dtype=np.uint8) 
        
        # Semantic: -1=None, >0=Instance ID
        grid_sem = np.full(self.map_size, -1, dtype=np.int32)
        
        # Metadata
        instances = {}
        
        # 1. Generate Rooms
        rooms = []
        num_rooms = random.randint(self.room_count_range[0], self.room_count_range[1])
        attempts = 0
        
        while len(rooms) < num_rooms and attempts < 100:
            attempts += 1
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            x = random.randint(20, self.map_size[1] - w - 20)
            y = random.randint(20, self.map_size[0] - h - 20)
            
            new_room = {'x': x, 'y': y, 'w': w, 'h': h, 'id': len(rooms) + 1}
            
            # Check overlap
            overlap = False
            pad = 10 # Spacing between rooms
            for r in rooms:
                if (x < r['x'] + r['w'] + pad and x + w + pad > r['x'] and
                    y < r['y'] + r['h'] + pad and y + h + pad > r['y']):
                    overlap = True
                    break
            
            if not overlap:
                rooms.append(new_room)
                # Carve room (Free Space)
                grid_geo[y:y+h, x:x+w] = 255
        
        if not rooms:
            return False # Failed to generate rooms

        # 2. Connect Rooms (Corridors)
        # Simple strategy: Connect room i to room i+1
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i+1]
            
            c1 = (r1['x'] + r1['w']//2, r1['y'] + r1['h']//2)
            c2 = (r2['x'] + r2['w']//2, r2['y'] + r2['h']//2)
            
            # Horizontal then Vertical
            x_min, x_max = min(c1[0], c2[0]), max(c1[0], c2[0])
            y_min, y_max = min(c1[1], c2[1]), max(c1[1], c2[1])
            
            # Draw L-shape corridor
            corridor_width = 25
            hw = corridor_width // 2
            
            # Horizontal segment
            grid_geo[c1[1]-hw:c1[1]+hw, x_min:x_max] = 255
            # Vertical segment
            grid_geo[y_min:y_max, c2[0]-hw:c2[0]+hw] = 255
            
            # Ensure connection point is clear
            if c1[0] < c2[0]: # Moving right
                grid_geo[c1[1]-hw:c1[1]+hw, x_max-hw:x_max+hw] = 255 # Corner
            else:
                grid_geo[c1[1]-hw:c1[1]+hw, x_min-hw:x_min+hw] = 255

        # 3. Place Objects
        instance_counter = 1
        for room in rooms:
            num_objs = random.randint(1, 3)
            for _ in range(num_objs):
                cat = random.choice(self.categories)
                
                # 1. Reduce Object Size (Parameter: obj_size_range)
                obj_w = random.randint(self.obj_size_range[0], self.obj_size_range[1])
                obj_h = random.randint(self.obj_size_range[0], self.obj_size_range[1])
                
                # Try to place in room
                for _ in range(20): # Increased attempts for better placement
                    # 2. Avoid Walls/Corners (Parameter: wall_padding)
                    min_x = room['x'] + self.wall_padding
                    max_x = room['x'] + room['w'] - obj_w - self.wall_padding
                    min_y = room['y'] + self.wall_padding
                    max_y = room['y'] + room['h'] - obj_h - self.wall_padding
                    
                    if min_x >= max_x or min_y >= max_y:
                        continue # Room too small for padding constraints
                        
                    ox = random.randint(min_x, max_x)
                    oy = random.randint(min_y, max_y)
                    
                    # 3. Avoid Clumping (Parameter: min_obj_spacing)
                    # Check a larger box around the proposed object location
                    check_x = max(0, ox - self.min_obj_spacing)
                    check_y = max(0, oy - self.min_obj_spacing)
                    check_w = obj_w + 2 * self.min_obj_spacing
                    check_h = obj_h + 2 * self.min_obj_spacing
                    
                    # Ensure check bounds are within map
                    check_x2 = min(self.map_size[1], check_x + check_w)
                    check_y2 = min(self.map_size[0], check_y + check_h)
                    
                    # Check if any existing object is in this expanded area
                    if np.any(grid_sem[check_y:check_y2, check_x:check_x2] != -1):
                        continue
                        
                    # Place Object
                    # Objects are OBSTACLES (0) in geometric map
                    grid_geo[oy:oy+obj_h, ox:ox+obj_w] = 0
                    
                    # Objects have ID in semantic map
                    inst_id = instance_counter
                    grid_sem[oy:oy+obj_h, ox:ox+obj_w] = inst_id
                    
                    # Record Instance
                    instances[str(inst_id)] = {
                        "category_name": cat["name"],
                        "room_id": room['id'],
                        "color_hex": cat["color"],
                        "raw_row": [str(inst_id), cat["color"], cat["name"], str(room['id'])]
                    }
                    
                    instance_counter += 1
                    break

        # 4. Prepare Output
        scene_name = f"scene_{scene_idx:03d}"
        scene_dir = os.path.join(self.output_base_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Save NPY
        np.save(os.path.join(scene_dir, "grid_geometric.npy"), grid_geo)
        np.save(os.path.join(scene_dir, "grid_semantic.npy"), grid_sem)
        
        # Save Metadata
        meta = {
            "scene_id": scene_name,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "resolution": self.resolution,
            "map_size_meters": [self.map_size[0] * self.resolution, self.map_size[1] * self.resolution],
            "map_pixel_size": list(self.map_size),
            "origin": [-10.0, -10.0, 0.0], # Arbitrary origin
            "floor_height": 0.0
        }
        with open(os.path.join(scene_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=4)
            
        with open(os.path.join(scene_dir, "instances.json"), 'w') as f:
            json.dump(instances, f, indent=4)
            
        # 5. Generate Visualizations
        # Vis Geometric
        vis_geo = grid_geo.copy() # Already 0/255
        cv2.imwrite(os.path.join(scene_dir, "vis_geometric.png"), vis_geo)
        
        # Vis Semantic
        vis_sem = np.full((self.map_size[0], self.map_size[1], 3), 200, dtype=np.uint8) # Gray bg
        # Draw Walls (Geometric 0) as Black
        vis_sem[grid_geo == 0] = [0, 0, 0]
        
        # Draw Objects
        for inst_id_str, info in instances.items():
            inst_id = int(inst_id_str)
            mask = (grid_sem == inst_id)
            r, g, b = self.hex_to_rgb(info['color_hex'])
            vis_sem[mask] = [b, g, r] # BGR for OpenCV
            
        cv2.imwrite(os.path.join(scene_dir, "vis_semantic.png"), vis_sem)
        
        return True

    def run(self):
        print(f"Generating {self.num_scenes} scenes in {self.output_base_dir}...")
        count = 0
        for i in range(self.num_scenes):
            if self.generate_single_scene(i):
                count += 1
                if count % 50 == 0:
                    print(f"Generated {count}/{self.num_scenes} scenes...")
            else:
                print(f"Failed to generate scene {i}, retrying...")
                # Simple retry logic could be added, but for now just skip/continue
        
        print(f"Completed! Generated {count} scenes.")

if __name__ == "__main__":
    # Easy: 2-3 rooms
    print("Generating Easy Dataset (2-3 rooms)...")
    generator_easy = MapGenerator(output_base_dir="generated_data_easy", num_scenes=500, room_count_range=(2, 3))
    generator_easy.run()

    # Medium: 4-5 rooms
    print("\nGenerating Medium Dataset (4-5 rooms)...")
    generator_medium = MapGenerator(output_base_dir="generated_data_medium", num_scenes=500, room_count_range=(4, 5))
    generator_medium.run()

    # Hard: 6-8 rooms
    print("\nGenerating Hard Dataset (6-8 rooms)...")
    generator_hard = MapGenerator(output_base_dir="generated_data_hard", num_scenes=500, room_count_range=(6, 8))
    generator_hard.run()
