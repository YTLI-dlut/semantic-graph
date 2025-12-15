import habitat_sim
import numpy as np
import os
import cv2
from habitat_mapper import HabitatMapper 

class ManualControlEnv:
    def __init__(self, scene_path):
        # 1. 配置仿真器
        self.sim_cfg = self.make_cfg(scene_path)
        self.sim = habitat_sim.Simulator(self.sim_cfg)
        self.agent = self.sim.initialize_agent(0)
        
        # 2. 初始化建图器 (地图范围 80米)
        self.mapper = HabitatMapper(map_size_meters=30.0, width=640, height=480)
        
        # 3. 动作映射
        self.key_mapping = {
            ord('w'): "move_forward",
            ord('a'): "turn_left",
            ord('d'): "turn_right",
        }
        
        print(f"Loaded Scene: {os.path.basename(scene_path)}")
        print("Controls: [W] Forward, [A] Left, [D] Right, [ESC] Quit")

    def make_cfg(self, scene_path):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = True

        sensor_specs = []
        
        # 深度相机
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [480, 640]
        depth_spec.position = [0.0, 0.88, 0.0]
        sensor_specs.append(depth_spec)

        # RGB 相机
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [480, 640]
        rgb_spec.position = [0.0, 0.88, 0.0]
        sensor_specs.append(rgb_spec)

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
            new_state = habitat_sim.AgentState()
            new_state.position = self.sim.pathfinder.get_random_navigable_point()
            import quaternion
            angle = np.random.uniform(0, 2*np.pi)
            axis = np.array([0, 1, 0])
            new_state.rotation = quaternion.from_rotation_vector(angle * axis)
            self.agent.set_state(new_state)
        
        self.mapper.reset()
        self.update_view(self.sim.get_sensor_observations())

    def update_view(self, obs):
        # 1. 获取数据
        depth = obs["depth_sensor"]
        rgb = obs["rgb_sensor"]
        
        # 2. 建图更新 (只负责计算 grid_map)
        agent_state = self.agent.get_state()
        self.mapper.update(depth, agent_state)
        
        # 3. [核心修改] 获取带红点和扇形的地图
        # 之前这里写的是 cv2.cvtColor(..., ...)，那是错误的，因为那是自己手动转，没有调用画红点的逻辑
        map_color = self.mapper.get_colored_map()
        
        # 4. 显示
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        cv2.imshow("First Person View (RGB)", rgb_bgr)
        cv2.imshow("Top-down Map (SDF)", map_color)

    def run(self):
        self.reset()
        print("Ready. Focusing on map window might help key response.")
        
        while True:
            key = cv2.waitKey(10) & 0xFF # 增加一点延时给 OpenCV 处理窗口事件
            
            if key == 27: 
                break
                
            if key in self.key_mapping:
                action = self.key_mapping[key]
                obs = self.sim.step(action)
                self.update_view(obs)

        self.sim.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你真实的场景路径
    scene_path = "/home/iiau/createGraph_ws/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    
    if os.path.exists(scene_path):
        env = ManualControlEnv(scene_path)
        env.run()
    else:
        print(f"Error: Scene path not found: {scene_path}")