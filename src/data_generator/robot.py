import habitat_sim
import numpy as np

class Robot:
    def __init__(self):
        self.agent_config = self.make_cfg()

    def make_cfg(self):
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # 1. RGB 相机
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [480, 640]
        rgb_spec.position = [0.0, 0.88, 0.0]
        
        # 2. 深度相机
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [480, 640]
        depth_spec.position = [0.0, 0.88, 0.0]
        
        # 3. 语义相机
        sem_spec = habitat_sim.CameraSensorSpec()
        sem_spec.uuid = "semantic_sensor"
        sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_spec.resolution = [480, 640]
        sem_spec.position = [0.0, 0.88, 0.0]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec, sem_spec]
        
        # === 动作空间配置 (新增 move_backward) ===
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        return agent_cfg