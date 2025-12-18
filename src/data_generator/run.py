import cv2
import numpy as np
import habitat_sim
from hm3d_env import HM3DEnvironment
from robot import Robot
from habitat_mapper import HabitatMapper
from control import KeyboardController

def draw_frame(env, mapper, obs):
    # 1. 获取数据
    rgb = obs["rgb_sensor"]
    depth = obs["depth_sensor"]
    semantic = obs["semantic_sensor"] # 获取语义数据
    
    # 2. 获取机器人状态
    agent_state = env.sim.get_agent(0).get_state()
    
    # 3. 更新地图
    mapper.update(depth, semantic, agent_state)
    
    # 4. 获取全尺寸原始地图 (可能是 1600x1600 像素，很大)
    geo_map_full = mapper.get_geometric_map_colored()
    sem_map_full = mapper.get_semantic_map_colored()
    
    # ==========================================
    # [核心修改] 局部地图裁剪逻辑 (Zoom In)
    # ==========================================
    
    # 设定：我们只想看机器人周围多少米的范围？
    local_view_meters = 20.0  # 只显示周围 20米 (可调整)
    
    # 计算裁剪窗口的像素大小 (20 / 0.05 = 400像素)
    crop_size = int(local_view_meters / mapper.resolution)
    half_crop = crop_size // 2
    
    if mapper.last_agent_pos is not None:
        # A. 计算机器人当前的像素坐标
        u_agent = int((mapper.last_agent_pos[0] / mapper.resolution) + mapper.map_center)
        v_agent = int((mapper.last_agent_pos[2] / mapper.resolution) + mapper.map_center)
        
        # B. 给原图加一圈边框 (Padding)
        # 作用：如果机器人走到地图最边缘，裁剪框会超出图片范围导致报错。
        # 加了边框后，就可以随意裁剪了。
        pad = half_crop
        geo_padded = cv2.copyMakeBorder(geo_map_full, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        sem_padded = cv2.copyMakeBorder(sem_map_full, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        
        # C. 计算裁剪坐标 (注意：因为加了pad，坐标要平移)
        center_u = u_agent + pad
        center_v = v_agent + pad
        
        # D. 执行裁剪
        geo_local = geo_padded[center_v-half_crop : center_v+half_crop, 
                               center_u-half_crop : center_u+half_crop]
        sem_local = sem_padded[center_v-half_crop : center_v+half_crop, 
                               center_u-half_crop : center_u+half_crop]
    else:
        # 如果还没有位置信息，就显示全图的中心
        h, w = geo_map_full.shape[:2]
        geo_local = geo_map_full[h//2-half_crop:h//2+half_crop, w//2-half_crop:w//2+half_crop]
        sem_local = sem_map_full[h//2-half_crop:h//2+half_crop, w//2-half_crop:w//2+half_crop]

    # ==========================================
    # [显示优化] 将裁剪出来的小图放大显示
    # ==========================================
    
    display_size = (600, 600) # 窗口大小
    
    # 使用最近邻插值放大 (INTER_NEAREST)，保证像素边缘清晰，不模糊
    geo_vis = cv2.resize(geo_local, display_size, interpolation=cv2.INTER_NEAREST)
    sem_vis = cv2.resize(sem_local, display_size, interpolation=cv2.INTER_NEAREST)
    
    # 处理 RGB 视图
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # 语义视图 (第一人称) 简单上色
    sem_1st_person = (semantic % 40) * (255 // 40) # 简易伪彩色
    sem_1st_person = cv2.applyColorMap(sem_1st_person.astype(np.uint8), cv2.COLORMAP_JET)

    # 显示所有窗口
    cv2.imshow("First Person RGB", rgb_bgr)
    # cv2.imshow("First Person Semantic", sem_1st_person) # 可选：显示第一人称语义
    
    cv2.imshow("Geometric Map (Local Nav)", geo_vis)
    cv2.imshow("Semantic Map (Local Instance)", sem_vis)

def main():
    # 场景 ID
    scene_id = "00800-TEEsavR23oF"
    
    # 初始化
    robot = Robot()
    env = HM3DEnvironment(scene_id, robot.agent_config)
    
    # 初始化建图器 (地图范围设大一点没关系，因为我们会局部裁剪)
    mapper = HabitatMapper(map_size_meters=80.0) 
    
    controller = KeyboardController()
    
    print("\n=== 系统就绪 ===")
    print("控制键: W(前) S(后) A(左) D(右) | ESC(退出)")
    print("地图模式: 局部跟随视角 (20m范围)")
    
    # 初始观测
    obs = env.reset()
    
    while True:
        # 绘图 & 建图
        draw_frame(env, mapper, obs)
        
        # 处理键盘输入
        key = cv2.waitKey(10)
        action = controller.get_action(key)
        
        if action == "quit":
            break
        elif action is not None:
            obs = env.step(action)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()