import numpy as np
import cv2
import quaternion # habitat-sim 自带的四元数库

class HabitatMapper:
    def __init__(self, 
                 map_size_meters=30.0,  # 地图物理尺寸 (米)
                 resolution=0.05,       # 分辨率 (米/像素)
                 width=640,             # 相机宽度
                 height=480,            # 相机高度
                 fov=90,                # 视场角
                 num_classes=80):       # [NEW] 语义类别数量
        
        # === 1. 地图参数 ===
        self.resolution = resolution
        self.map_size_pixels = int(map_size_meters / resolution)
        self.map_center = self.map_size_pixels // 2
        self.map_size_meters = map_size_meters
        self.num_classes = num_classes

        self.min_x = -map_size_meters / 2.0
        self.min_z = -map_size_meters / 2.0
        
        # [地图 1] 几何地图: 127=未知, 255=空闲(White), 0=障碍(Black)
        self.grid_map = np.full((self.map_size_pixels, self.map_size_pixels), 127, dtype=np.uint8)
        
        # [地图 2] 概率语义计数: (H, W, K) 存储置信度累积
        self.semantic_counts = np.zeros((self.map_size_pixels, self.map_size_pixels, num_classes), dtype=np.float32)
        
        # [地图 2 - 缓存] 语义地图: 缓存 ArgMax 结果, 初始化为 -1
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
        self.semantic_counts.fill(0)
        self.semantic_map.fill(-1)
        self.last_agent_pos = None
        self.last_agent_rot = None

    def update(self, depth_obs, semantic_obs, agent_state, check_is_floor_callback=None, confidence_obs=None):
        """
        核心更新函数：同时更新几何地图和语义地图
        :param depth_obs: 深度图 (H, W)
        :param semantic_obs: 语义图 (H, W), 存储 Instance ID
        :param agent_state: 智能体状态 (包含 position, rotation)
        :param check_is_floor_callback: (可选) 函数, 输入 instance_id 返回 bool
        :param confidence_obs: (可选) 置信度图 (H, W), 对应 semantic_obs 每个像素的置信度
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
        
        # [关键] 提取对应的语义 ID 和置信度
        sem_ids_valid = semantic_obs[mask] 
        if confidence_obs is None:
            conf_values = np.ones_like(sem_ids_valid, dtype=np.float32)
        else:
            conf_values = confidence_obs[mask]

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
        
        # 判定逻辑 (几何)
        is_obstacle = (rel_height > self.min_height_rel) & (rel_height < self.max_height_rel)
        is_ground = (rel_height <= self.min_height_rel) & (rel_height > -0.5)
        
        # [NEW] 判定逻辑 (语义辅助)
        if check_is_floor_callback is not None:
            unique_ids = np.unique(sem_ids_valid)
            floor_ids = []
            for uid in unique_ids:
                if check_is_floor_callback(int(uid)):
                    floor_ids.append(uid)
            
            is_semantic_floor = np.isin(sem_ids_valid, floor_ids)
            is_ground[is_semantic_floor] = True
            is_obstacle[is_semantic_floor] = False

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
        sem_ids_final = sem_ids_valid[valid_indices] 
        conf_final = conf_values[valid_indices]
        
        # === 6. 更新数据 ===
        
        # [Map 1] 更新几何地图
        self.grid_map[v[is_ground], u[is_ground]] = 255
        self.grid_map[v[is_obstacle], u[is_obstacle]] = 0
        
        # [Map 2] 更新语义概率计数
        # 只记录障碍物区域的语义信息
        if np.sum(is_obstacle) > 0:
            obs_v = v[is_obstacle]
            obs_u = u[is_obstacle]
            obs_ids = sem_ids_final[is_obstacle]
            obs_conf = conf_final[is_obstacle]
            
            # 使用 np.add.at 进行原地累加 (Handling hash collisions/duplicates in same batch)
            # 需要过滤无效 ID ( -1 )
            valid_id_mask = (obs_ids >= 0) & (obs_ids < self.num_classes)
            
            if np.any(valid_id_mask):
                np.add.at(self.semantic_counts, (obs_v[valid_id_mask], obs_u[valid_id_mask], obs_ids[valid_id_mask]), obs_conf[valid_id_mask])
                
                # 可选：更新缓存的 ArgMax Map (只更新本次变动的区域以节省时间，或按需全量计算)
                # 这里为了简单，暂不每次都全量ArgMax，等到 get_semantic_map 时再算
                # 或者：只更新当前观测到的区域
                # current_max_ids = np.argmax(self.semantic_counts[obs_v, obs_u], axis=-1)
                # self.semantic_map[obs_v, obs_u] = current_max_ids

    def get_geometric_map_colored(self, draw_agent=True):
        """ 获取可视化的几何地图 (带智能体位置) """
        color_map = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)
        
        if not draw_agent or self.last_agent_pos is None:
            return color_map

        u_agent = int((self.last_agent_pos[0] / self.resolution) + self.map_center)
        v_agent = int((self.last_agent_pos[2] / self.resolution) + self.map_center)
        
        if not (0 <= u_agent < self.map_size_pixels and 0 <= v_agent < self.map_size_pixels):
            return color_map

        # 绘制视野扇形
        overlay = color_map.copy()
        rot_mat = quaternion.as_rotation_matrix(self.last_agent_rot)
        forward = rot_mat @ np.array([0, 0, -1])
        angle_deg = np.rad2deg(np.arctan2(forward[2], forward[0]))
        radius = int(self.max_dist / self.resolution)
        
        cv2.ellipse(overlay, (u_agent, v_agent), (radius, radius), 
                    angle_deg, -self.fov/2, self.fov/2, (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, color_map, 0.6, 0, color_map)
        cv2.circle(color_map, (u_agent, v_agent), 5, (0, 0, 255), -1)
        
        return color_map

    def get_semantic_map_colored(self, draw_agent=True):
        """ 获取可视化的语义地图 (根据概率分布 ArgMax) """
        # 1. 实时计算 ArgMax
        # 注意：为了性能，如果地图巨大，这一步会慢。
        # 优化：只计算被观测过的区域 (sum > 0)
        
        total_counts = np.sum(self.semantic_counts, axis=-1)
        observed_mask = total_counts > 0
        
        vis_map = np.full((self.map_size_pixels, self.map_size_pixels, 3), 127, dtype=np.uint8)
        
        # 2. 绘制 Free 区域 (白色)
        vis_map[self.grid_map == 255] = [255, 255, 255]
        
        # 3. 绘制语义物体 (仅在有观测的地方)
        if np.any(observed_mask):
            # 获取最大概率的 ID
            best_ids = np.argmax(self.semantic_counts, axis=-1)
            
            ids = best_ids[observed_mask]
            
            # 伪彩色
            r = (ids * 13 + 50) % 255
            g = (ids * 47 + 80) % 255
            b = (ids * 101 + 110) % 255
            
            vis_map[observed_mask] = np.stack([b, g, r], axis=-1)
            
        # 4. 绘制智能体
        if draw_agent and self.last_agent_pos is not None:
            u = int((self.last_agent_pos[0] / self.resolution) + self.map_center)
            v = int((self.last_agent_pos[2] / self.resolution) + self.map_center)
            if 0 <= u < self.map_size_pixels and 0 <= v < self.map_size_pixels:
                cv2.circle(vis_map, (u, v), 5, (0, 0, 255), -1)

        return vis_map

    def get_entropy_map_colored(self):
        """ 计算并可视化熵地图 (热力图) """
        # H = - sum(p * log(p))
        epsilon = 1e-6
        
        total_counts = np.sum(self.semantic_counts, axis=-1, keepdims=True)
        valid_mask = (total_counts > 0).squeeze()
        
        entropy_map = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.float32)
        
        if np.any(valid_mask):
            probs = self.semantic_counts[valid_mask] / (total_counts[valid_mask] + epsilon)
            entropy = -np.sum(probs * np.log2(probs + epsilon), axis=-1)
            entropy_map[valid_mask] = entropy
            
        # 归一化用于可视化 (0 - MaxEntropy)
        # MaxEntropy 对于 N 类是 log2(N)
        max_entropy = np.log2(self.num_classes) if self.num_classes > 1 else 1.0
        norm_entropy = (entropy_map / max_entropy * 255).astype(np.uint8)
        
        # 应用热力图颜色映射 (Jet: Blue=Low, Red=High)
        heatmap = cv2.applyColorMap(norm_entropy, cv2.COLORMAP_JET)
        
        # 背景设为黑色或灰色以免混淆
        # heatmap[~valid_mask] = [0, 0, 0] # 已经是0了
        
        return heatmap