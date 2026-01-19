
import os
import json
import numpy as np
import cv2
import random

class DataLoader:
    def __init__(self, data_dir="output/manual_maps"):
        self.data_dir = data_dir
        self.map_list = self._scan_maps()
        print(f"[DataLoader] Found {len(self.map_list)} valid maps in {data_dir}")

    def _scan_maps(self):
        """扫描所有包含完整数据的地图文件夹"""
        valid_maps = []
        if not os.path.exists(self.data_dir):
            return []
            
        for d in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, d)
            if not os.path.isdir(full_path):
                continue
                
            # 检查必要文件
            geo_path = os.path.join(full_path, "grid_geometric.npy")
            sem_path = os.path.join(full_path, "grid_semantic.npy")
            meta_path = os.path.join(full_path, "metadata.json")
            
            if os.path.exists(geo_path) and os.path.exists(sem_path) and os.path.exists(meta_path):
                valid_maps.append(full_path)
                
        valid_maps.sort()
        return valid_maps

    def get_random_map(self):
        if not self.map_list:
            raise ValueError("No maps found!")
        path = random.choice(self.map_list)
        return self.load_map(path)

    def load_map(self, map_path):
        """
        加载并预处理地图为 MAME 格式
        MAME 格式: 0=Free, 1=Obstacle, 127/Other=Unknown (在 env 中处理)
        Habitat 格式: 255=Free, 0=Obstacle, 127=Unknown
        """
        # 加载数据
        geo_map = np.load(os.path.join(map_path, "grid_geometric.npy"))
        sem_map = np.load(os.path.join(map_path, "grid_semantic.npy"))
        with open(os.path.join(map_path, "metadata.json"), 'r') as f:
            meta = json.load(f)
            
        # 预处理 Geometric Map
        # 目标: Free(0), Obstacle(1), Unknown(127)
        # 原始: Free(255), Obstacle(0), Unknown(127)
        
        normalized_map = np.full_like(geo_map, 127)
        normalized_map[geo_map == 255] = 255 # Free
        normalized_map[geo_map == 0] = 1   # Obstacle
        
        # 裁剪掉多余的 Padding
        rows, cols = np.where(normalized_map != 127)
        if len(rows) > 0:
            r_min, r_max = np.min(rows), np.max(rows)
            c_min, c_max = np.min(cols), np.max(cols)
            
            # Add padding
            pad = 20
            r_min = max(0, r_min - pad)
            r_max = min(normalized_map.shape[0], r_max + pad)
            c_min = max(0, c_min - pad)
            c_max = min(normalized_map.shape[1], c_max + pad)
            
            normalized_map = normalized_map[r_min:r_max, c_min:c_max]
            sem_map = sem_map[r_min:r_max, c_min:c_max]
        
        return {
            "geometric": normalized_map,
            "semantic": sem_map,
            "metadata": meta,
            "path": map_path
        }

if __name__ == "__main__":
    loader = DataLoader()
    if loader.map_list:
        data = loader.get_random_map()
        print(f"Loaded map: {data['path']}")
        print(f"Geo Shape: {data['geometric'].shape}")
        print(f"Sem Shape: {data['semantic'].shape}")
        
