import numpy as np
import cv2
import quaternion # habitat-sim 自带

class HabitatMapper:
    def __init__(self, 
                 map_size_meters=40.0, # [修改] 改大一点 (80米)，防止跑出界
                 resolution=0.05, 
                 width=640, 
                 height=480, 
                 fov=90):
        
        # === 1. 地图参数 ===
        self.resolution = resolution
        self.map_size_pixels = int(map_size_meters / resolution)
        self.map_center = self.map_size_pixels // 2
        
        # 初始化地图: 127=未知, 255=空闲(White), 0=障碍(Black)
        self.grid_map = np.full((self.map_size_pixels, self.map_size_pixels), 127, dtype=np.uint8)
        
        # 可视化状态缓存
        self.last_agent_pos = None
        self.last_agent_rot = None
        self.fov = fov
        self.map_size_meters = map_size_meters
        
        # === 2. 相机内参 ===
        self.width = width
        self.height = height
        self.fx = width / (2 * np.tan(np.deg2rad(fov) / 2))
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # 预计算像素网格
        u = np.arange(self.width)
        v = np.arange(self.height)
        self.uu, self.vv = np.meshgrid(u, v)
        
        # 预计算因子
        self.factor_x = (self.uu - self.cx) / self.fx
        self.factor_y = (self.vv - self.cy) / self.fy
        
        # === 3. 过滤参数 ===
        self.min_height_rel = 0.10
        self.max_height_rel = 1.8
        self.max_dist = 4.0
        self.sensor_height = 0.88 

    def reset(self):
        self.grid_map.fill(127)
        self.last_agent_pos = None
        self.last_agent_rot = None

    def update(self, depth_obs, agent_state):
        # 1. 无条件更新位姿
        self.last_agent_pos = agent_state.position
        self.last_agent_rot = agent_state.rotation
        
        # 2. 获取位姿
        pos = agent_state.position 
        rot = agent_state.rotation
        
        # 3. 深度图过滤
        mask = (depth_obs > 0.1) & (depth_obs < self.max_dist)
        if np.count_nonzero(mask) == 0:
            return self.grid_map

        # 4. 反投影
        z_c = -depth_obs[mask] 
        x_c = -self.factor_x[mask] * z_c 
        y_c = self.factor_y[mask] * z_c 
        
        points_cam = np.stack([x_c, y_c, z_c], axis=1)
        
        # 5. 转世界坐标
        rot_mat = quaternion.as_rotation_matrix(rot)
        points_world = points_cam @ rot_mat.T + pos
        points_world[:, 1] += self.sensor_height
        
        # 6. 投影到 2D 栅格
        px = points_world[:, 0]
        py = points_world[:, 1]
        pz = points_world[:, 2]
        
        rel_height = py - pos[1]
        
        # 判定逻辑
        is_obstacle = (rel_height > self.min_height_rel) & (rel_height < self.max_height_rel)
        is_ground = (rel_height <= self.min_height_rel) & (rel_height > -0.5)
        
        # 坐标映射
        u = ((px / self.resolution) + self.map_center).astype(np.int32)
        v = ((pz / self.resolution) + self.map_center).astype(np.int32)
        
        # 边界检查
        valid_indices = (u >= 0) & (u < self.map_size_pixels) & \
                        (v >= 0) & (v < self.map_size_pixels)
        
        u = u[valid_indices]
        v = v[valid_indices]
        is_obstacle = is_obstacle[valid_indices]
        is_ground = is_ground[valid_indices]
        
        # 填色
        self.grid_map[v[is_ground], u[is_ground]] = 255
        self.grid_map[v[is_obstacle], u[is_obstacle]] = 0
        
        return self.grid_map

    def get_colored_map(self):
        color_map = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)
        
        if self.last_agent_pos is None:
            return color_map

        # 1. 计算像素位置 (u, v)
        # u 对应 x, v 对应 z
        u_agent = int((self.last_agent_pos[0] / self.resolution) + self.map_center)
        v_agent = int((self.last_agent_pos[2] / self.resolution) + self.map_center)
        
        # 2. [新增] 越界检查与打印
        if not (0 <= u_agent < self.map_size_pixels and 0 <= v_agent < self.map_size_pixels):
            # 如果你在终端看到这句话，说明你的地图太小了，或者出生点太偏了
            print(f"\r[Warning] Agent out of map! Pos: {self.last_agent_pos}, MapIdx: ({u_agent}, {v_agent})", end="")
            return color_map

        # 3. 绘制半透明扇形
        overlay = color_map.copy()
        
        # 计算朝向
        rot_mat = quaternion.as_rotation_matrix(self.last_agent_rot)
        forward = rot_mat @ np.array([0, 0, -1])
        
        # 角度计算
        angle_deg = np.rad2deg(np.arctan2(forward[2], forward[0]))
        radius = int(self.max_dist / self.resolution)
        
        cv2.ellipse(overlay, 
                    (u_agent, v_agent), 
                    (radius, radius), 
                    angle_deg, 
                    -self.fov / 2, 
                    self.fov / 2, 
                    (0, 255, 255), 
                    -1)
        
        cv2.addWeighted(overlay, 0.4, color_map, 0.6, 0, color_map)
        
        # 4. 绘制红色圆点
        cv2.circle(color_map, (u_agent, v_agent), 5, (0, 0, 255), -1)
        
        return color_map