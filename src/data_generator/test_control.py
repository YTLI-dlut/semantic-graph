
import habitat_sim
import cv2
import random
import os

# 尝试连接当前存在的 X Server
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":1"

import imageio
import numpy as np

# 配置场景路径
test_scene = "data/scene_datasets/hm3d/val/00806-tQ5s4ShP627/tQ5s4ShP627.basis.glb"

def make_simple_cfg(settings):
    # 配置后端
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = False
    
    # 显式指定 GPU 设备 ID，通常 0 是第一个 GPU
    sim_cfg.gpu_device_id = 0 

    # 配置 Agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # RGB 传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]
    
    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def main():
    if not os.path.exists(test_scene):
        print(f"Error: 场景文件不存在: {test_scene}")
        return

    print("初始化仿真环境 (Headless Mode)...")

    # 1. 初始化仿真器
    cfg = make_simple_cfg({
        "scene": test_scene,
        "width": 640,
        "height": 480
    })
    sim = habitat_sim.Simulator(cfg)

    # 2. 初始化智能体位置
    agent = sim.initialize_agent(0)
    if sim.pathfinder.is_loaded:
        start_state = habitat_sim.AgentState()
        start_state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(start_state)
        print(f"Agent 初始化位置: {start_state.position}")
    else:
        print("Warning: 场景没有 NavMesh，无法随机导航，使用默认原点")

    # 3. 自动动作序列
    actions = ["move_forward", "turn_left", "turn_right"]
    frames = []
    
    steps = 30
    print(f"开始执行 {steps} 步随机漫游...")

    try:
        for i in range(steps):
            # 随机选择动作
            action = random.choice(actions)
            sim.step(action)
            
            # 获取观测
            observations = sim.get_sensor_observations()
            rgb = observations["color_sensor"]
            
            # 去除 Alpha 通道 (如果存在)
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            
            frames.append(rgb)
            print(f"Step {i+1}/{steps}: {action}")

        # 4. 保存 GIF
        output_path = "habitat_test_walk.gif"
        print(f"正在保存 GIF 到: {output_path}")
        imageio.mimsave(output_path, frames, duration=0.1) # 10 fps
        print("✅ 完成！")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        sim.close()

if __name__ == "__main__":
    main()
