import habitat_sim
import numpy as np
import os
import cv2
from habitat_mapper import HabitatMapper # 导入上面写的类

class HabitatTrainingEnv:
    def __init__(self, scene_path, headless=False):
        # === 1. 配置仿真器 ===
        self.sim_cfg = self.make_cfg(scene_path)
        self.sim = habitat_sim.Simulator(self.sim_cfg)
        self.agent = self.sim.initialize_agent(0)
        
        # === 2. 初始化建图器 ===
        # 注意: 这里的 width/height 必须和 make_cfg 里的一致
        self.mapper = HabitatMapper(map_size_meters=40.0, width=640, height=480)
        
        # === 3. 定义动作 ===
        self.action_mapping = {
            0: "move_forward",
            1: "turn_left",
            2: "turn_right"
        }
        
        # 可视化窗口 (如果是 headless 训练模式就关掉)
        self.headless = headless
        
        print(f"Loaded Scene: {scene_path}")

    def make_cfg(self, scene_path):
        # 仿照 run_habitat.py 的配置，但去掉了 ROS 部分
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = True # 如果不需要物理碰撞，设为 False 跑得更快

        sensor_specs = []
        
        # 深度相机
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [480, 640] # H, W
        depth_spec.position = [0.0, 0.88, 0.0]
        depth_spec.orientation = [0.0, 0.0, 0.0]
        sensor_specs.append(depth_spec)

        # RGB 相机
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [480, 640]
        rgb_spec.position = [0.0, 0.88, 0.0]
        rgb_spec.orientation = [0.0, 0.0, 0.0] 
        sensor_specs.append(rgb_spec)
        
        # 语义相机 (为训练提供真值)
        sem_spec = habitat_sim.CameraSensorSpec()
        sem_spec.uuid = "semantic_sensor"
        sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_spec.resolution = [480, 640]
        sem_spec.position = [0.0, 0.88, 0.0]
        sem_spec.orientation = [0.0, 0.0, 0.0]
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
        # 随机重置 Agent 位置
        if self.sim.pathfinder.is_loaded:
            new_state = habitat_sim.AgentState()
            new_state.position = self.sim.pathfinder.get_random_navigable_point()
            # 随机旋转
            import quaternion
            angle = np.random.uniform(0, 2*np.pi)
            axis = np.array([0, 1, 0]) # 绕 Y 轴
            new_state.rotation = quaternion.from_rotation_vector(angle * axis)
            self.agent.set_state(new_state)
        
        self.mapper.reset()
        
        # 获取第一帧观测
        obs = self.sim.get_sensor_observations()
        return self._process_obs(obs)

    def step(self, action_idx):
        """
        执行一步动作
        action_idx: 0 (前), 1 (左), 2 (右)
        """
        action_name = self.action_mapping.get(action_idx, "move_forward")
        
        # 执行动作
        obs = self.sim.step(action_name)
        
        # 处理数据
        state_dict = self._process_obs(obs)
        
        # 这里你可以计算 reward 和 done
        reward = 0 
        done = False
        
        return state_dict, reward, done, {}

    def _process_obs(self, obs):
        # 1. 获取传感器数据
        depth = obs["depth_sensor"]
        rgb = obs["rgb_sensor"]
        semantic = obs["semantic_sensor"] # 真值 (H, W) int32
        
        # 2. 获取 Agent 真实位姿
        agent_state = self.agent.get_state()
        
        # 3. 建图 (极速版)
        current_map = self.mapper.update(depth, agent_state)
        
        # 4. (可选) 可视化 debug
        if not self.headless:
            cv2.imshow("Map", current_map)
            # cv2.imshow("RGB", rgb[..., :3]) # 去掉 alpha 通道
            cv2.waitKey(1)
            
        return {
            "rgb": rgb,
            "depth": depth,
            "semantic": semantic, # 这是 Object ID，你需要映射成 Class ID
            "map": current_map,   # 2D 栅格图 (可以直接喂给 RL)
            "pose": agent_state.position
        }

    def close(self):
        self.sim.close()

# === 测试代码 ===
if __name__ == "__main__":
    # 替换为你自己的场景路径
    scene_path = "/home/iiau/createGraph_ws/data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    
    # 检查文件是否存在
    if not os.path.exists(scene_path):
        print(f"Error: Scene not found at {scene_path}")
    else:
        env = HabitatTrainingEnv(scene_path, headless=False)
        env.reset()
        
        print("Start moving... Press 'q' in map window to exit if blocked.")
        
        for i in range(100):
            # 随机游走: 80% 前进, 10% 左转, 10% 右转
            action = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
            env.step(action)
            
        env.close()