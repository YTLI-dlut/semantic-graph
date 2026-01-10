
import argparse
import os
import time
import numpy as np
import habitat_sim
import quaternion
from hm3d_env import HM3DEnvironment
from robot import Robot
from habitat_mapper import HabitatMapper
from utils import SemanticParser, get_floor_heights, try_teleport_to_floor, save_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=int, default=None, help="Target floor index (0-based)")
    parser.add_argument("--fixed_height", type=float, default=None, help="Manually specify target floor height (meters)")
    parser.add_argument("--save_size", type=int, default=1000, help="Fixed map size for saving (default: 1000x1000)")
    parser.add_argument("--scene_id", type=str, default="00800-TEEsavR23oF", help="HM3D Scene ID")
    parser.add_argument("--samples", type=int, default=2000, help="Number of teleport samples for auto mapping")
    args = parser.parse_args()

    scene_id = args.scene_id
    
    # 初始化
    robot = Robot()
    env = HM3DEnvironment(scene_id, robot.agent_config)
    
    # 初始化 Mapper
    map_size_meters = 80.0
    resolution = 0.05
    mapper = HabitatMapper(map_size_meters=map_size_meters, resolution=resolution)
    
    # 初始化语义解析器
    parser = SemanticParser(scene_id)
    
    # 楼层处理
    target_floor_height = None
    floor_idx = 0
    
    if args.fixed_height is not None:
        target_floor_height = args.fixed_height
        floor_idx = "custom"
        print(f"Using manually specified floor height: {target_floor_height:.2f}")
    else:
        floors = get_floor_heights(env)
        if args.floor is not None:
            if 0 <= args.floor < len(floors):
                target_floor_height = floors[args.floor]
                floor_idx = args.floor
                print(f"Selected Floor {args.floor}: Height {target_floor_height:.2f}")
            else:
                print(f"Error: Floor index {args.floor} out of range (0-{len(floors)-1}). Using default.")
                if len(floors) > 0: target_floor_height = floors[0]
        else:
            if len(floors) > 0: target_floor_height = floors[0]
            
    # Reset
    obs = env.reset()
    mapper.reset()
    
    # 初始 teleport
    if target_floor_height is not None:
        print(f"Teleporting to floor {target_floor_height:.2f}...")
        try_teleport_to_floor(env, target_floor_height)
        
    print(f"=== Auto Floor Mapper Started (Target: Floor {floor_idx} @ {target_floor_height:.2f}m) ===")
    print(f"Running {args.samples} samples...")
    
    # 自动建图循环
    pathfinder = env.sim.pathfinder
    agent = env.sim.get_agent(0)
    
    # 地面语义回调 (复用逻辑)
    def check_is_floor(inst_id):
        info = parser.get_info(inst_id)
        if info is None: return False
        name = info["category_name"].lower()
        floor_keywords = ["floor", "carpet", "rug", "mat", "tile", "wood", "ground", "pavement"]
        for kw in floor_keywords:
            if kw in name:
                return True
        return False
    
    count_success = 0
    start_time = time.time()
    
    # 优化：提前获取 pathfinder 引用和 helper
    
    for i in range(args.samples):
        # 1. 随机采样点
        p = pathfinder.get_random_navigable_point()
        
        # 2. 检查该点是否在目标楼层 (宽松判定，因为我们只关心这一层的点)
        if target_floor_height is not None and abs(p[1] - target_floor_height) > 0.3:
            continue
            
        # 3. 瞬移
        state = habitat_sim.AgentState()
        state.position = p
        angle = np.random.uniform(0, 2*np.pi)
        state.rotation = quaternion.from_rotation_vector([0, angle, 0])
        agent.set_state(state)
        
        # 4. 获取观测 & 更新地图
        obs = env.sim.get_sensor_observations()
        depth = obs["depth_sensor"]
        semantic = obs["semantic_sensor"]
        mapper.update(depth, semantic, state, check_is_floor_callback=check_is_floor)
        
        count_success += 1
        if i % 100 == 0:
            print(f"Processed {i}/{args.samples} samples (Valid: {count_success})")
            
    elapsed = time.time() - start_time
    print(f"Done. Processed {count_success} valid scans in {elapsed:.2f}s.")
    
    # 保存
    meta_info = {
        "scene_id": scene_id,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "resolution": resolution,
        "map_size_meters": map_size_meters,
        "map_pixel_size": mapper.grid_map.shape if hasattr(mapper, "grid_map") else [],
        "origin": [
            mapper.min_x if hasattr(mapper, "min_x") else -map_size_meters/2,
            mapper.min_z if hasattr(mapper, "min_z") else -map_size_meters/2,
            0
        ], 
        "floor_height": float(target_floor_height) if target_floor_height else 0.0
    }
    
    out_dir = f"output/manual_maps/{scene_id}_floor{floor_idx}"
    save_dataset(out_dir, mapper, parser, meta_info, save_size=args.save_size)
    
    if os.path.exists(out_dir):
        os.utime(out_dir, None)
        
    env.close()

if __name__ == "__main__":
    main()
