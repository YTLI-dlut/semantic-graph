import habitat_sim

class RobotController:
    def __init__(self, camera_height=1.0):
        self.camera_height = camera_height
        # 定义传感器分辨率
        self.resolution = [480, 640]

    def get_agent_config(self):
        """ 生成 Habitat Agent 配置，包含更细腻的动作空间 """
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # --- 1. 定义传感器 (RGB, Depth, Semantic) ---
        sensors = []
        for sensor_type, uuid in [
            (habitat_sim.SensorType.COLOR, "rgb_sensor"),
            (habitat_sim.SensorType.DEPTH, "depth_sensor"),
            (habitat_sim.SensorType.SEMANTIC, "semantic_sensor"),
        ]:
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = sensor_type
            spec.resolution = self.resolution
            spec.position = [0.0, self.camera_height, 0.0]
            sensors.append(spec)
        
        agent_cfg.sensor_specifications = sensors

        # --- 2. 定义细腻的动作空间 ---
        # 步长: 0.1米 (之前是0.25米) -> 移动更平滑
        # 转角: 5度 (之前是10度或30度) -> 转向更精准
        move_step = 0.1
        turn_angle = 5.0
        
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=move_step)
            ),
            # Habitat 默认没有 backward，我们复用 move_forward 但给负的 amount
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=-move_step)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_angle)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_angle)
            ),
        }
        return agent_cfg

    def get_key_mapping(self):
        """ 键盘控制映射 """
        return {
            ord('w'): "move_forward",
            ord('s'): "move_backward", # 新增后退
            ord('a'): "turn_left",
            ord('d'): "turn_right",
        }