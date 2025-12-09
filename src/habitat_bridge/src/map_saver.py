#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

class MapToImageSaver:
    def __init__(self):
        rospy.init_node('map_saver_py', anonymous=True)
        
        # === 1. 参数配置 (必须与 launch 文件一致) ===
        # 地图物理尺寸 (米)
        self.map_size_x = 40.0 
        self.map_size_y = 40.0 
        # 分辨率 (米/像素)
        self.resolution = 0.05 
        
        # 计算图片像素大小
        self.width = int(self.map_size_x / self.resolution)
        self.height = int(self.map_size_y / self.resolution)
        
        # 地图原点 (世界坐标系下的左下角)
        # exploration_node 默认把 (0,0) 设在地图中心
        self.origin_x = -self.map_size_x / 2.0
        self.origin_y = -self.map_size_y / 2.0
        
        # === 2. 初始化画布 ===
        # 默认为 127 (灰色/未知)
        self.map_img = np.full((self.height, self.width), 127, dtype=np.uint8)
        
        # 创建保存文件夹
        self.save_dir = "python_map_dataset"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.save_count = 0
        
        # === 3. 订阅话题 ===
        # 注意：这里订阅的是 PointCloud2
        rospy.Subscriber("/grid_map/free", PointCloud2, self.free_cb)
        # rospy.Subscriber("/grid_map/occupied", PointCloud2, self.occupied_cb)
        rospy.Subscriber("/grid_map/occupied_inflate", PointCloud2, self.occupied_cb)
        
        # 定时保存 (例如每 1 秒保存一次，或者根据你的 step 逻辑触发)
        rospy.Timer(rospy.Duration(1.0), self.save_map)
        
        print(f"Map Saver Initialized. Image Size: {self.width}x{self.height}")

    def world_to_pixel(self, x, y):
        """ 将世界坐标 (x, y) 转换为 图片像素坐标 (u, v) """
        u = int((x - self.origin_x) / self.resolution)
        v = int((y - self.origin_y) / self.resolution)
        
        # 坐标系翻转：图像原本 (0,0) 在左上角，世界坐标 y 向上
        # 所以 v 需要翻转
        v = self.height - 1 - v
        
        return u, v

    def process_cloud(self, msg, color_value):
        """ 解析点云并在画布上填色 """
        # 使用 point_cloud2 读取生成器 (比手动解析快)
        gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        
        for p in gen:
            x, y = p[0], p[1]
            
            # 过滤掉地图范围外的点
            if x < self.origin_x or x >= -self.origin_x or \
               y < self.origin_y or y >= -self.origin_y:
                continue
                
            u, v = self.world_to_pixel(x, y)
            
            # 边界检查
            if 0 <= u < self.width and 0 <= v < self.height:
                self.map_img[v, u] = color_value

    def free_cb(self, msg):
        # 收到空闲点云 -> 填白色 (255)
        # 这里的策略是：只有当该像素不是障碍物时才更新
        # 但通常直接覆盖即可，或者先画 Free 再画 Occupied
        self.process_cloud(msg, 255)

    def occupied_cb(self, msg):
        # 收到占据点云 -> 填黑色 (0)
        # Occupied 优先级最高，直接覆盖
        self.process_cloud(msg, 0)

    def save_map(self, event):
        filename = os.path.join(self.save_dir, f"step_{self.save_count}.png")
        cv2.imwrite(filename, self.map_img)
        self.save_count += 1
        # print(f"Saved: {filename}")

if __name__ == '__main__':
    try:
        saver = MapToImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass