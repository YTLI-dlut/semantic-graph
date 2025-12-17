import habitat_sim
import numpy as np
import os
import cv2
from habitat_mapper import HabitatMapper 

class ManualControlEnv:
    def __init__(self, dataset_config_file, scene_handle):
        # ================= 参数配置区域 =================
        # 1. 统一相机高度 (1.0米 是个折中值，盲区小，视野也不错)
        self.camera_height = 1.0 
        # 2. 地图范围
        self.map_size = 40.0
        # ===============================================

        # 1. 启动仿真器
        print("--- 正在启动仿真器 ---")
        self.sim_cfg = self.make_cfg(dataset_config_file, scene_handle)
        
        try:
            self.sim = habitat_sim.Simulator(self.sim_cfg)
        except Exception as e:
            print(f"[错误] 启动失败: {e}")
            exit()

        self.agent = self.sim.initialize_agent(0)
        
        # 2. 初始化建图器
        self.mapper = HabitatMapper(map_size_meters=self.map_size, width=640, height=480)
        
        # 【核心修复】将统一的高度参数传给建图器
        # 只有这里和 sim 配置一致，地面才不会算错！
        self.mapper.sensor_height = self.camera_height
        
        # 3. 按键映射
        self.key_mapping = {
            ord('w'): "move_forward",
            ord('a'): "turn_left",
            ord('d'): "turn_right",
        }
        
        print("\n=== 就绪 ===")
        print(f"相机高度: {self.camera_height}m (已同步到建图算法)")
        print("操作: [W/A/D] 移动, [ESC] 退出")

    def make_cfg(self, config_file, scene_id):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_dataset_config_file = config_file
        sim_cfg.scene_id = scene_id
        sim_cfg.enable_physics = True

        sensor_specs = []
        RESOLUTION = [480, 640]
        
        # 使用 self.camera_height 确保统一
        POS = [0.0, self.camera_height, 0.0] 
        
        # RGB
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = RESOLUTION
        rgb_spec.position = POS
        sensor_specs.append(rgb_spec)

        # Depth
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = RESOLUTION
        depth_spec.position = POS
        sensor_specs.append(depth_spec)

        # Semantic
        sem_spec = habitat_sim.CameraSensorSpec()
        sem_spec.uuid = "semantic_sensor"
        sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_spec.resolution = RESOLUTION
        sem_spec.position = POS
        sensor_specs.append(sem_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
            "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
            "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
        }
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def reset(self):
        if self.sim.pathfinder.is_loaded:
            self.agent.set_state(habitat_sim.AgentState(position=self.sim.pathfinder.get_random_navigable_point()))
        self.mapper.reset()
        self.update_view(self.sim.get_sensor_observations())

    def update_view(self, obs):
        depth = obs["depth_sensor"]
        rgb = obs["rgb_sensor"]
        semantic = obs["semantic_sensor"].astype(np.int32)
        
        self.mapper.update(depth, self.agent.get_state())
        map_color = self.mapper.get_colored_map()
        
        h, w = semantic.shape
        center_id = semantic[h // 2, w // 2]
        
        # 获取语义名称
        obj_name = "Unknown"
        scene = self.sim.semantic_scene
        if scene and len(scene.objects) > 0:
            if center_id < len(scene.objects):
                 obj = scene.objects[center_id]
                 if obj and obj.category:
                     obj_name = obj.category.name()
        
        # 语义可视化
        np.random.seed(42) 
        palette = np.random.randint(0, 255, (np.max(semantic)+100, 3), dtype=np.uint8)
        palette[0] = [20, 20, 20]
        safe_semantic = np.clip(semantic, 0, len(palette)-1)
        sem_vis = palette[safe_semantic]
        
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        cv2.line(rgb_bgr, (w//2-10, h//2), (w//2+10, h//2), (0, 255, 0), 2)
        cv2.line(rgb_bgr, (w//2, h//2-10), (w//2, h//2+10), (0, 255, 0), 2)
        cv2.putText(rgb_bgr, f"ID: {center_id} | {obj_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("RGB", rgb_bgr)
        cv2.imshow("Semantic", sem_vis)
        cv2.imshow("Map", map_color)

    def run(self):
        self.reset()
        while True:
            k = cv2.waitKey(10) & 0xFF
            if k == 27: break
            
            action = None
            if k == ord('w'): action = "move_forward"
            elif k == ord('a'): action = "turn_left"
            elif k == ord('d'): action = "turn_right"
            
            if action:
                self.update_view(self.sim.step(action))
                
        self.sim.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CONFIG_FILE = "/home/iiau/HM3D/hm3d_unified_val/hm3d_annotated_val_basis.scene_dataset_config.json"
    SCENE_HANDLE = "00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    
    if os.path.exists(CONFIG_FILE):
        env = ManualControlEnv(CONFIG_FILE, SCENE_HANDLE)
        env.run()
    else:
        print(f"错误: 找不到配置文件 {CONFIG_FILE}")