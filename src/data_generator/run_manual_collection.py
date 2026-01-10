
import os
import argparse
import time
import json
import csv
import cv2
import numpy as np
import habitat_sim
import quaternion
from hm3d_env import HM3DEnvironment
from robot import Robot
from habitat_mapper import HabitatMapper
from control import KeyboardController
from utils import SemanticParser, get_floor_heights, try_teleport_to_floor, save_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=int, default=None, help="Target floor index (0-based)")
    parser.add_argument("--fixed_height", type=float, default=None, help="Manually specify target floor height (meters)")
    parser.add_argument("--save_size", type=int, default=1000, help="Fixed map size for saving (default: 1000x1000)")
    parser.add_argument("--scene_id", type=str, default="00800-TEEsavR23oF", help="HM3D Scene ID")
    args = parser.parse_args()

    scene_id = args.scene_id
    
    # 初始化
    robot = Robot()
    env = HM3DEnvironment(scene_id, robot.agent_config)
    
    # 初始化 Mapper
    map_size_meters = 80.0
    resolution = 0.05
    mapper = HabitatMapper(map_size_meters=map_size_meters, resolution=resolution)
    
    # 初始化控制器
    controller = KeyboardController()
    
    # 初始化语义解析器
    parser = SemanticParser(scene_id)
    
    # 楼层处理
    target_floor_height = None
    
    if args.fixed_height is not None:
        target_floor_height = args.fixed_height
        print(f"Using manually specified floor height: {target_floor_height:.2f}")
    else:
        floors = get_floor_heights(env)
        if args.floor is not None:
            if 0 <= args.floor < len(floors):
                target_floor_height = floors[args.floor]
                print(f"Selected Floor {args.floor}: Height {target_floor_height:.2f}")
            else:
                print(f"Error: Floor index {args.floor} out of range (0-{len(floors)-1}). Using default.")
                if len(floors) > 0: target_floor_height = floors[0]

            
    # Reset
    obs = env.reset()
    mapper.reset()
    
    # 如果指定了楼层，瞬移过去
    if target_floor_height is not None:
        success = try_teleport_to_floor(env, target_floor_height)
        if not success:
            raise RuntimeError(f"Could not teleport agent to target floor height {target_floor_height}!")
        obs = env.sim.get_sensor_observations() # 更新观测
        
    print("=== Manual Collection Started ===")
    print("Controls: W/A/S/D to move, P/Q to Save & Quit, ESC to Quit without saving.")
    
    # 获取初始高度作为楼层参考
    agent_pos = env.sim.get_agent(0).get_state().position
    floor_height = agent_pos[1]
    
    while True:
        # 1. 显示当前地图
        geo_img = mapper.get_geometric_map_colored(draw_agent=True)
        cv2.imshow("Manual Mapping - Geometric", geo_img)
        
        # 2. 显示 RGB 视角
        if "rgb_sensor" in obs:
            rgb_obs = obs["rgb_sensor"]
            # Habitat 返回的可能是 RGBA 或 RGB
            if rgb_obs.shape[2] == 4:
                bgr_img = cv2.cvtColor(rgb_obs, cv2.COLOR_RGBA2BGR)
            else:
                bgr_img = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
            cv2.imshow("Agent View", bgr_img)
        
        # 处理输入
        key = cv2.waitKey(10)
        action_name = controller.get_action(key)
        
        if action_name == "quit":
            print("Quitting without saving...")
            break
        elif action_name == "save_and_quit":
            # 准备元数据
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
                "floor_height": float(floor_height)
            }
            
            # [Updated] Naming: scene_id_floorX
            # Try to determine floor index
            floor_idx = 0
            if args.floor is not None:
                floor_idx = args.floor
            elif args.fixed_height is not None:
                floor_idx = "custom"
            
            # Construct path
            out_dir = f"output/manual_maps/{scene_id}_floor{floor_idx}"
            
            save_dataset(out_dir, mapper, parser, meta_info, save_size=args.save_size)
            
            # Update directory modification time so verification scripts find it
            if os.path.exists(out_dir):
                os.utime(out_dir, None)
                
            break
        elif action_name == "jump":
            # 实现跳跃/瞬移逻辑
            # 1. 获取当前状态
            agent = env.sim.get_agent(0)
            state = agent.get_state()
            # 2. 计算前方 1.5 米的点
            # state.rotation 是四元数，我们需要前向向量
            # habitat 中 -Z 是前方
            forward_local = np.array([0, 0, -1.5]) 
            rot_mat = quaternion.as_rotation_matrix(state.rotation)
            forward_world = rot_mat @ forward_local
            target_pos = state.position + forward_world
            
            # 3. Snap 到 NavMesh
            pathfinder = env.sim.pathfinder
            snapped_pos = pathfinder.snap_point(target_pos)
            snapped_pos = np.array(snapped_pos)
            
            if not np.isnan(snapped_pos[0]):
                state.position = snapped_pos
                agent.set_state(state)
                obs = env.sim.get_sensor_observations() # 瞬移后立即更新观测
                print(f"Jumped to {snapped_pos}")
                
                # 更新地图 (可选，跳跃后通常也想更新一下)
                depth = obs["depth_sensor"]
                semantic = obs["semantic_sensor"]
                mapper.update(depth, semantic, state)
            else:
                print("Jump failed: Target not navigable.")
            
        if action_name in ["move_forward", "move_backward", "turn_left", "turn_right"]:
            obs = env.step(action_name)
            
            # 更新地图
            depth = obs["depth_sensor"]
            semantic = obs["semantic_sensor"]
            agent_state = env.sim.get_agent(0).get_state()
            
            # 定义地面检测回调
            def check_is_floor(inst_id):
                info = parser.get_info(inst_id)
                if info is None: return False
                name = info["category_name"].lower()
                # 关键词白名单
                floor_keywords = ["floor", "carpet", "rug", "mat", "tile", "wood", "ground", "pavement"]
                for kw in floor_keywords:
                    if kw in name:
                        return True
                return False
            
            mapper.update(depth, semantic, agent_state, check_is_floor_callback=check_is_floor)
            
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
