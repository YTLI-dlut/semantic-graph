
import os
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

class SemanticParser:
    def __init__(self, scene_id, semantic_root_dir="/home/iiau/HM3D/hm3d-val-semantic-annots-v0.2"):
        """
        解析 .semantic.txt 文件
        格式参考: 27,2D1BBC,"window",1
        (InstanceID, ColorHex, CategoryName, RoomID)
        """
        self.semantic_map = {}
        txt_path = os.path.join(semantic_root_dir, scene_id, f"{scene_id.split('-')[-1]}.semantic.txt")
        
        if not os.path.exists(txt_path):
            print(f"[Warning] Semantic info file not found: {txt_path}")
            return

        print(f"Parsing semantic file: {txt_path}")
        with open(txt_path, 'r') as f:
            reader = csv.reader(f)
            # 跳过可能的 header? 通常 HM3D txt 第一行就是数据，或者有 header。
            # 根据用户提供的片段，直接是数据。我们尝试检测一下。
            for row in reader:
                if len(row) < 4: continue
                try:
                    # 尝试转换 ID
                    inst_id = int(row[0])
                    color_hex = row[1]
                    cat_name = row[2]
                    room_id = int(row[3])
                    
                    self.semantic_map[inst_id] = {
                        "category_name": cat_name,
                        "room_id": room_id,
                        "color_hex": color_hex,
                        "raw_row": row
                    }
                except ValueError:
                    continue
        print(f"Loaded {len(self.semantic_map)} semantic instances.")

    def get_info(self, inst_id):
        return self.semantic_map.get(inst_id, None)

def save_dataset(output_dir, mapper, semantic_parser, meta_info, save_size=1000):
    """保存最终的地图和元数据 (固定尺寸居中裁剪/填充)"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving dataset to {output_dir} (Size: {save_size}x{save_size})...")
    
    # 获取原始地图引用
    raw_geo = getattr(mapper, "grid_map", None)
    raw_sem = getattr(mapper, "semantic_map", None)
    
    # === 1.5 [NEW] 封闭未知区域边界 ===
    # 目的：确保 Free 区域被 Obstacle 包围，不直接与 Unknown 接触
    # 算法：
    # 1. 提取 Free 掩码
    # 2. 膨胀 Free 掩码
    # 3. 找到 (原本是 Unknown) 且 (膨胀后被覆盖) 的区域 -> 设为 Obstacle
    
    # 127: Unknown, 255: Free, 0: Obstacle
    free_mask = (raw_geo == 255).astype(np.uint8)
    
    # 膨胀 kernel 3x3
    kernel = np.ones((3, 3), np.uint8)
    dilated_free = cv2.dilate(free_mask, kernel, iterations=1)
    
    # 找出边界上的未知点
    # 条件: 原本是 127 (Unknown) AND 膨胀后的 Free 掩码覆盖到了
    boundary_mask = (raw_geo == 127) & (dilated_free == 1)
    
    if np.count_nonzero(boundary_mask) > 0:
        print(f"Sealing map boundaries: marked {np.count_nonzero(boundary_mask)} pixels as obstacles.")
        raw_geo[boundary_mask] = 0 # 强制设为障碍物
        # 语义地图对应位置也可以设为某种特殊 ID (例如 0 或 background)，保持为 -1 也没关系
        # 但既然是障碍物，HabitatMapper 逻辑通常会分配语义。
        # 这里我们保持 semantic map 为 -1 或者设为 0 (wall/structure通常ID较低?)
        # 简单起见，不改语义地图，或者设为 0
        
    # === 1. 找到有效区域边界 ===
    valid_mask = (raw_geo != 127)
    
    if np.count_nonzero(valid_mask) == 0:
        print("[Warning] Map is completely empty. Saving empty fixed map.")
        # 中心点默认为地图中心
        center_u = raw_geo.shape[1] // 2
        center_v = raw_geo.shape[0] // 2
    else:
        # 获取有效区域的边界
        rows = np.any(valid_mask, axis=1)
        cols = np.any(valid_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 计算有效区域的中心
        center_v = (rmin + rmax) // 2
        center_u = (cmin + cmax) // 2
        
    # === 2. 计算固定尺寸裁剪区域 ===
    # 我们希望 (center_v, center_u) 位于新图的中心
    half_size = save_size // 2
    
    # 原始图中需要提取的范围 (可能越界)
    src_v_min = center_v - half_size
    src_v_max = center_v + half_size
    src_u_min = center_u - half_size
    src_u_max = center_u + half_size
    
    # 目标图中需要填充的范围
    dst_v_min = 0
    dst_v_max = save_size
    dst_u_min = 0
    dst_u_max = save_size
    
    # 计算重叠区域 (Intersection)
    # 也就是实际要把 原始图的哪一块 copy 到 目标图的哪一块
    
    # 处理 V (行/高度)
    inter_v_min_src = max(0, src_v_min)
    inter_v_max_src = min(raw_geo.shape[0], src_v_max)
    
    offset_v = inter_v_min_src - src_v_min # 如果 src 越过上边界，dst 要往下偏移
    inter_v_min_dst = dst_v_min + offset_v
    inter_v_len = inter_v_max_src - inter_v_min_src
    inter_v_max_dst = inter_v_min_dst + inter_v_len
    
    # 处理 U (列/宽度)
    inter_u_min_src = max(0, src_u_min)
    inter_u_max_src = min(raw_geo.shape[1], src_u_max)
    
    offset_u = inter_u_min_src - src_u_min
    inter_u_min_dst = dst_u_min + offset_u
    inter_u_len = inter_u_max_src - inter_u_min_src
    inter_u_max_dst = inter_u_min_dst + inter_u_len
    
    # === 3. 创建输出 Buffer并填充 ===
    # Geometric init with 127
    out_geo = np.full((save_size, save_size), 127, dtype=np.uint8)
    # Semantic init with -1
    out_sem = np.full((save_size, save_size), -1, dtype=np.int32)
    
    # 执行 Copy
    if inter_v_len > 0 and inter_u_len > 0:
        out_geo[inter_v_min_dst:inter_v_max_dst, inter_u_min_dst:inter_u_max_dst] = \
            raw_geo[inter_v_min_src:inter_v_max_src, inter_u_min_src:inter_u_max_src]
            
        out_sem[inter_v_min_dst:inter_v_max_dst, inter_u_min_dst:inter_u_max_dst] = \
            raw_sem[inter_v_min_src:inter_v_max_src, inter_u_min_src:inter_u_max_src]
            
    # 保存 .npy
    np.save(os.path.join(output_dir, "grid_geometric.npy"), out_geo)
    np.save(os.path.join(output_dir, "grid_semantic.npy"), out_sem)
    
    # === 4. 可视化保存 ===
    full_vis_geo = mapper.get_geometric_map_colored(draw_agent=False)
    full_vis_sem = mapper.get_semantic_map_colored(draw_agent=False)
    
    out_vis_geo = np.full((save_size, save_size, 3), 127, dtype=np.uint8)
    out_vis_sem = np.full((save_size, save_size, 3), 127, dtype=np.uint8)
    
    if inter_v_len > 0 and inter_u_len > 0:
        out_vis_geo[inter_v_min_dst:inter_v_max_dst, inter_u_min_dst:inter_u_max_dst] = \
            full_vis_geo[inter_v_min_src:inter_v_max_src, inter_u_min_src:inter_u_max_src]
            
        out_vis_sem[inter_v_min_dst:inter_v_max_dst, inter_u_min_dst:inter_u_max_dst] = \
            full_vis_sem[inter_v_min_src:inter_v_max_src, inter_u_min_src:inter_u_max_src]

    cv2.imwrite(os.path.join(output_dir, "vis_geometric.png"), out_vis_geo)
    cv2.imwrite(os.path.join(output_dir, "vis_semantic.png"), out_vis_sem)

    # === 5. 更新元数据 ===
    # 原始 Origin 对应 raw_geo 的 (0,0)
    # 新图的 (0,0) 对应 raw_geo 的 (src_u_min, src_v_min)
    # 即使 src_u_min 是负数也没关系，Origin 公式依然成立
    
    old_origin = meta_info["origin"] 
    res = meta_info["resolution"]
    
    new_origin = [
        old_origin[0] + src_u_min * res,
        old_origin[1] + src_v_min * res,
        old_origin[2]
    ]
    
    meta_info["origin"] = new_origin
    meta_info["map_pixel_size"] = [save_size, save_size]
    meta_info["map_size_meters"] = [save_size * res, save_size * res]
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(meta_info, f, indent=4)
        
    # === 6. 生成 instances.json ===
    unique_ids = np.unique(out_sem)
    instances_dict = {}
    for uid in unique_ids:
        uid = int(uid)
        if uid == -1 or uid == 0: continue
        info = semantic_parser.get_info(uid)
        if info:
            instances_dict[uid] = info
        else:
            instances_dict[uid] = {"category_name": "unknown", "room_id": -1}
            
    with open(os.path.join(output_dir, "instances.json"), 'w') as f:
        json.dump(instances_dict, f, indent=4)
        
    print(f"Save complete. Fixed Size: {save_size}x{save_size}")

import argparse

def get_floor_heights(env, num_samples=50000, bin_size=0.10):
    """
    自动分析场景中的楼层高度 (Histogram Peak Detection)
    之前的简单聚类会被楼梯误导，导致将 1F 和 2F 合并为中间层。
    直方图方法寻找采样点密度最大的高度（即平坦的地面），过滤掉稀疏的楼梯。
    """
    print("Analyzing scene floor heights (Histogram Method)...")
    pathfinder = env.sim.pathfinder
    points = []
    
    # 1. 大规模采样
    for _ in range(num_samples):
        points.append(pathfinder.get_random_navigable_point())
    
    points = np.array(points)
    if len(points) == 0:
        return []

    heights = points[:, 1]
    h_min, h_max = np.min(heights), np.max(heights)
    print(f"  [Debug] Sampled {num_samples} points. Height range: [{h_min:.2f}, {h_max:.2f}]")

    # 2. 直方图统计
    # 动态计算 bins 数量
    num_bins = int((h_max - h_min) / bin_size) + 1
    hist, bin_edges = np.histogram(heights, bins=num_bins, range=(h_min, h_max))
    
    # 3. 峰值检测
    # 过滤掉低密度区域（例如楼梯）
    # 阈值：最大峰值的 5%
    max_count = np.max(hist)
    threshold = max_count * 0.05
    
    valid_indices = np.where(hist > threshold)[0]
    
    floor_heights = []
    if len(valid_indices) > 0:
        # 简单聚类：相邻的高密度 bin 视为同一层，取加权平均或由于 bin 很小直接取中心
        # 这里我们合并相邻 bin
        current_cluster_bins = [valid_indices[0]]
        
        for idx in valid_indices[1:]:
            if idx == current_cluster_bins[-1] + 1: # 连续
                current_cluster_bins.append(idx)
            else:
                # 结算上一组
                # 计算加权平均高度
                cluster_counts = hist[current_cluster_bins]
                cluster_centers = (bin_edges[current_cluster_bins] + bin_edges[np.array(current_cluster_bins)+1]) / 2.0
                weighted_h = np.average(cluster_centers, weights=cluster_counts)
                floor_heights.append(weighted_h)
                
                # 新起一组
                current_cluster_bins = [idx]
        
        # 结算最后一组
        cluster_counts = hist[current_cluster_bins]
        cluster_centers = (bin_edges[current_cluster_bins] + bin_edges[np.array(current_cluster_bins)+1]) / 2.0
        weighted_h = np.average(cluster_centers, weights=cluster_counts)
        floor_heights.append(weighted_h)
    
    floor_heights = np.array(floor_heights)
    # 排序
    floor_heights.sort()
    
    print(f"Found {len(floor_heights)} floors at heights: {np.round(floor_heights, 2)}")
    return floor_heights

def try_teleport_to_floor(env, floor_height):
    """尝试将智能体传送到指定高度附近的有效位置 (Adaptive Strategy)"""
    pathfinder = env.sim.pathfinder
    agent = env.sim.get_agent(0)
    
    # 策略配置: (最大尝试次数, 高度阈值, 是否检查平坦度, 平坦度阈值)
    strategies = [
        {"name": "Strict",   "attempts": 2000, "h_tol": 0.15, "check_flat": True,  "flat_tol": 0.10},
        {"name": "Relaxed",  "attempts": 2000, "h_tol": 0.25, "check_flat": True,  "flat_tol": 0.20},
        {"name": "Fallback", "attempts": 1000, "h_tol": 0.30, "check_flat": False, "flat_tol": 99.9},
    ]
    
    for strategy in strategies:
        print(f"Trying teleport with {strategy['name']} strategy...")
        
        for i in range(strategy["attempts"]):
            p = pathfinder.get_random_navigable_point()
            
            # 1. 高度检查
            if abs(p[1] - floor_height) > strategy["h_tol"]:
                continue
                
            # 2. 平坦度检测
            is_flat = True
            if strategy["check_flat"]:
                num_neighbors = 5
                check_radius = 1.0 # 米
                
                for _ in range(num_neighbors):
                    # 随机偏移
                    offset = np.random.uniform(-check_radius, check_radius, size=3)
                    offset[1] = 0 #只在水平面偏移
                    neighbor_target = p + offset
                    
                    # Snap 到最近的可导航点
                    neighbor_snapped = pathfinder.snap_point(neighbor_target)
                    neighbor_snapped = np.array(neighbor_snapped) 
                    
                    dist_horizontal = np.linalg.norm(neighbor_snapped[[0, 2]] - neighbor_target[[0, 2]])
                    
                    if not np.isnan(neighbor_snapped[0]) and dist_horizontal < 0.5:
                        height_diff = abs(neighbor_snapped[1] - p[1])
                        if height_diff > strategy["flat_tol"]: 
                            is_flat = False
                            break
            
            if is_flat:
                state = habitat_sim.AgentState()
                state.position = p
                # 随机朝向
                angle = np.random.uniform(0, 2*np.pi)
                state.rotation = quaternion.from_rotation_vector([0, angle, 0])
                agent.set_state(state)
                print(f"Teleported to floor {floor_height:.2f} at {p} (Strategy: {strategy['name']})")
                return True
                
    # 如果所有策略都失败
    print(f"[Error] Failed to teleport to floor {floor_height:.2f} even with fallback strategy!")
    return False

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
