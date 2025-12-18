import numpy as np
import cv2
import quaternion # habitat-sim 自带的四元数库

class HabitatMapper:
    def __init__(self, 
                 map_size_meters=30.0,  # 地图物理尺寸 (米)
                 resolution=0.05,       # 分辨率 (米/像素)
                 width=640,             # 相机宽度
                 height=480,            # 相机高度
                 fov=90):               # 视场角
        
        # === 1. 地图参数 ===
        self.resolution = resolution
        self.map_size_pixels = int(map_size_meters / resolution)
        self.map_center = self.map_size_pixels // 2
        self.map_size_meters = map_size_meters
        
        # [地图 1] 几何地图: 127=未知, 255=空闲(White), 0=障碍(Black)
        self.grid_map = np.full((self.map_size_pixels, self.map_size_pixels), 127, dtype=np.uint8)
        
        # [地图 2] 语义地图: 存储 Instance ID, 初始化为 -1 (代表无物体/未知)
        self.semantic_map = np.full((self.map_size_pixels, self.map_size_pixels), -1, dtype=np.int32)
        
        # 可视化状态缓存
        self.last_agent_pos = None
        self.last_agent_rot = None
        self.fov = fov
        
        # === 2. 相机内参预计算 ===
        self.width = width
        self.height = height
        self.fx = width / (2 * np.tan(np.deg2rad(fov) / 2))
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # 预计算像素网格 (u, v)
        u = np.arange(self.width)
        v = np.arange(self.height)
        self.uu, self.vv = np.meshgrid(u, v)
        
        # 预计算反投影因子
        self.factor_x = (self.uu - self.cx) / self.fx
        self.factor_y = (self.vv - self.cy) / self.fy
        
        # === 3. 过滤参数 (根据机器人高度调整) ===
        self.min_height_rel = 0.10   # 障碍物最小高度 (相对于脚底)
        self.max_height_rel = 1.8    # 障碍物最大高度 (忽略天花板)
        self.max_dist = 4.0          # 深度图最大有效距离
        self.sensor_height = 0.88    # 传感器距地面高度

    def reset(self):
        """ 重置所有地图 """
        self.grid_map.fill(127)
        self.semantic_map.fill(-1)
        self.last_agent_pos = None
        self.last_agent_rot = None

    def update(self, depth_obs, semantic_obs, agent_state):
        """
        核心更新函数：同时更新几何地图和语义地图
        :param depth_obs: 深度图 (H, W)
        :param semantic_obs: 语义图 (H, W), 存储 Instance ID
        :param agent_state: 智能体状态 (包含 position, rotation)
        """
        # 1. 记录位姿用于可视化
        self.last_agent_pos = agent_state.position
        self.last_agent_rot = agent_state.rotation
        
        pos = agent_state.position 
        rot = agent_state.rotation
        
        # 2. 深度图有效性掩码
        mask = (depth_obs > 0.1) & (depth_obs < self.max_dist)
        if np.count_nonzero(mask) == 0:
            return
        
        # 3. 反投影 (2D -> 3D Camera Coords)
        z_c = -depth_obs[mask] 
        x_c = -self.factor_x[mask] * z_c 
        y_c = self.factor_y[mask] * z_c 
        
        points_cam = np.stack([x_c, y_c, z_c], axis=1)
        
        # [关键] 提取对应的语义 ID
        sem_ids_valid = semantic_obs[mask] # 只取有效深度点对应的语义ID
        
        # 4. 转世界坐标 (Camera -> World)
        rot_mat = quaternion.as_rotation_matrix(rot)
        points_world = points_cam @ rot_mat.T + pos
        points_world[:, 1] += self.sensor_height # 加上相机高度
        
        # 5. 投影到 2D 栅格地图 (World -> Grid Map)
        px = points_world[:, 0]
        py = points_world[:, 1]
        pz = points_world[:, 2]
        
        # 计算相对高度 (用于判断是地面还是障碍物)
        rel_height = py - pos[1]
        
        # 判定逻辑
        is_obstacle = (rel_height > self.min_height_rel) & (rel_height < self.max_height_rel)
        is_ground = (rel_height <= self.min_height_rel) & (rel_height > -0.5)
        
        # 坐标离散化
        u = ((px / self.resolution) + self.map_center).astype(np.int32)
        v = ((pz / self.resolution) + self.map_center).astype(np.int32)
        
        # 边界检查
        valid_indices = (u >= 0) & (u < self.map_size_pixels) & \
                        (v >= 0) & (v < self.map_size_pixels)
        
        # 筛选出地图范围内的点
        u = u[valid_indices]
        v = v[valid_indices]
        is_obstacle = is_obstacle[valid_indices]
        is_ground = is_ground[valid_indices]
        sem_ids_final = sem_ids_valid[valid_indices] # 保持同步
        
        # === 6. 更新数据 ===
        
        # [Map 1] 更新几何地图
        # 规则：先画 Free (255)，再画 Obstacle (0) 覆盖
        self.grid_map[v[is_ground], u[is_ground]] = 255
        self.grid_map[v[is_obstacle], u[is_obstacle]] = 0
        
        # [Map 2] 更新语义地图
        # 规则：只记录障碍物区域的语义信息 (因为我们关心的是“那个物体是什么”)
        # 忽略地面的语义 (通常是 floor 或 carpet)，保持地图清晰
        if np.sum(is_obstacle) > 0:
            self.semantic_map[v[is_obstacle], u[is_obstacle]] = sem_ids_final[is_obstacle]

    def get_geometric_map_colored(self):
        """ 获取可视化的几何地图 (带智能体位置) """
        # 转为 BGR
        color_map = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)
        
        if self.last_agent_pos is None:
            return color_map

        # 计算智能体在地图上的坐标
        u_agent = int((self.last_agent_pos[0] / self.resolution) + self.map_center)
        v_agent = int((self.last_agent_pos[2] / self.resolution) + self.map_center)
        
        # 越界保护
        if not (0 <= u_agent < self.map_size_pixels and 0 <= v_agent < self.map_size_pixels):
            return color_map

        # 绘制视野扇形 (半透明)
        overlay = color_map.copy()
        rot_mat = quaternion.as_rotation_matrix(self.last_agent_rot)
        forward = rot_mat @ np.array([0, 0, -1])
        angle_deg = np.rad2deg(np.arctan2(forward[2], forward[0]))
        radius = int(self.max_dist / self.resolution)
        
        cv2.ellipse(overlay, (u_agent, v_agent), (radius, radius), 
                    angle_deg, -self.fov/2, self.fov/2, (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, color_map, 0.6, 0, color_map)
        
        # 绘制红色位置点
        cv2.circle(color_map, (u_agent, v_agent), 5, (0, 0, 255), -1)
        
        return color_map

    def get_semantic_map_colored(self):
        """ 获取可视化的语义地图 (随机彩色) """
        # 1. 创建背景 (灰色)
        vis_map = np.full((self.map_size_pixels, self.map_size_pixels, 3), 127, dtype=np.uint8)
        
        # 2. 绘制 Free 区域 (白色) - 基于几何地图
        vis_map[self.grid_map == 255] = [255, 255, 255]
        
        # 3. 绘制语义物体
        # 提取所有有效的语义像素 (ID != -1)
        valid_mask = self.semantic_map > -1
        ids = self.semantic_map[valid_mask]
        
        if len(ids) > 0:
            # 使用 Hash 算法生成伪随机颜色，保证同一个 ID 颜色永远固定
            # 公式: (ID * Prime + Offset) % 255
            r = (ids * 13 + 50) % 255
            g = (ids * 47 + 80) % 255
            b = (ids * 101 + 110) % 255
            
            # 填色 (OpenCV 使用 BGR 顺序)
            vis_map[valid_mask] = np.stack([b, g, r], axis=-1)
            
        # 4. 绘制智能体位置 (方便对照)
        if self.last_agent_pos is not None:
            u = int((self.last_agent_pos[0] / self.resolution) + self.map_center)
            v = int((self.last_agent_pos[2] / self.resolution) + self.map_center)
            if 0 <= u < self.map_size_pixels and 0 <= v < self.map_size_pixels:
                cv2.circle(vis_map, (u, v), 5, (0, 0, 255), -1)

        return vis_map