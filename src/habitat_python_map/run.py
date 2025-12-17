import cv2
import numpy as np
from hm3d_env import HM3DEnvironment
from robot import RobotController
from habitat_mapper import HabitatMapper # 确保你的 mapper 文件也在同级目录

# 配置场景
SCENE_ID = "00800-TEEsavR23oF"

def main():
    # 1. 初始化控制器 (设定相机高度 1.0米)
    robot = RobotController(camera_height=1.0)
    
    # 2. 初始化环境 (传入机器人的配置)
    env = HM3DEnvironment(SCENE_ID, robot.get_agent_config())
    
    # 3. 初始化建图器 (同步相机高度!)
    mapper = HabitatMapper(map_size_meters=40.0, width=640, height=480)
    mapper.sensor_height = robot.camera_height # 关键：参数同步
    
    # 4. 准备交互
    key_map = robot.get_key_mapping()
    print("\n=== 系统就绪 ===")
    print("控制键: W(前) S(后) A(左) D(右) | ESC(退出)")
    
    # 初始观测
    obs = env.reset()
    
    while True:
        # --- 可视化渲染逻辑 ---
        draw_frame(env, mapper, obs)
        
        # --- 控制逻辑 ---
        k = cv2.waitKey(10) & 0xFF
        if k == 27: break # ESC
        
        if k in key_map:
            action = key_map[k]
            obs = env.step(action)

    env.close()
    cv2.destroyAllWindows()

def draw_frame(env, mapper, obs):
    """ 渲染 RGB、语义和地图 """
    # 1. 获取数据
    rgb = cv2.cvtColor(obs["rgb_sensor"], cv2.COLOR_RGBA2BGR)
    depth = obs["depth_sensor"]
    semantic = obs["semantic_sensor"].astype(np.int32)
    
    # 2. 更新并获取地图
    # 获取 Agent 状态 (绝对坐标)
    agent_state = env.sim.get_agent(0).get_state()
    mapper.update(depth, agent_state)
    map_color = mapper.get_colored_map()
    
    # 3. 语义可视化 (中心点射线检测)
    h, w = semantic.shape
    center_id = semantic[h // 2, w // 2]
    
    obj_name = "Unknown"
    # 使用官方 API 查找名字 (因为已经正确加载了 json 配置)
    try:
        scene = env.sim.semantic_scene
        if center_id < len(scene.objects):
            obj = scene.objects[center_id]
            if obj and obj.category:
                obj_name = obj.category.name()
    except: pass

    # 伪彩色语义图
    np.random.seed(42)
    palette = np.random.randint(0, 255, (np.max(semantic)+100, 3), dtype=np.uint8)
    palette[0] = [20, 20, 20]
    sem_vis = palette[semantic]
    
    # UI 绘制
    cv2.line(rgb, (w//2-10, h//2), (w//2+10, h//2), (0, 255, 0), 2)
    cv2.line(rgb, (w//2, h//2-10), (w//2, h//2+10), (0, 255, 0), 2)
    cv2.putText(rgb, f"ID: {center_id} | {obj_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示
    cv2.imshow("RGB View", rgb)
    cv2.imshow("Semantic View", sem_vis)
    cv2.imshow("2D Map", map_color)

if __name__ == "__main__":
    main()