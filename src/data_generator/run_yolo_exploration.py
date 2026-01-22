
import os
import argparse
import time
import json
import cv2
import numpy as np
import habitat_sim
import quaternion
from ultralytics import YOLOWorld
import random
import imageio

# 引入项目模块
from robot import Robot
from habitat_mapper import HabitatMapper
from control import KeyboardController

def main():
    if "DISPLAY" not in os.environ:
        print("\n [Error] DISPLAY environment variable not found!")
        print(" To see the window and control the robot, you must enable SSH X11 Forwarding.")
        print(" 1. Disconnect current session.")
        print(" 2. Reconnect using: ssh -X user@host")
        print(" 3. (Windows users) Ensure VcXsrv or Xming is running.")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, default="00800-TEEsavR23oF", help="HM3D Scene ID")
    parser.add_argument("--model", type=str, default="models/yolov8s-world.pt", help="YOLO-World model path")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps for auto exploration")
    args = parser.parse_args()

    # ================= 0. 目录与路径准备 =================
    # timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/yolo_runs/{args.scene_id}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True) # 确保模型目录存在
    
    print(f"Output directory: {output_dir}")

    # ================= 1. 动态加载提示词 (Classes) =================
    # 尝试从 instances.json 加载真实物体类别
    default_classes = ["floor", "wall", "chair", "table", "sofa", "bed", "tv", "person", "window", "door"]
    classes = []
    
    json_path = f"output/manual_maps/{args.scene_id}_floor0/instances.json"
    if os.path.exists(json_path):
        print(f"Loading instance names from {json_path}...")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # 遍历字典提取 category_name
                categories = set()
                for _, item in data.items():
                    if "category_name" in item:
                        categories.add(item["category_name"])
                
                classes = sorted(list(categories))
                print(f"Found {len(classes)} unique categories.")
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            classes = default_classes
    else:
        print(f"Instance JSON not found at {json_path}. Using default classes.")
        classes = default_classes
        
    # 如果列表为空 (以防万一)，使用默认
    if not classes:
        classes = default_classes
        
    # 始终确保 floor 和 wall 存在 (用于回调逻辑)
    if "floor" not in classes: classes.insert(0, "floor")
    if "wall" not in classes: classes.insert(1, "wall")

    # 保存配置
    config_data = vars(args)
    config_data["classes"] = classes
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    # ================= 2. 初始化 YOLO-World =================
    print(f"Loading YOLO-World model: {args.model}...")
    # 如果 path 不包含 /，假设在 models/ 下
    if os.sep not in args.model:
        model_path = os.path.join("models", args.model)
    else:
        model_path = args.model
        
    model = YOLOWorld(model_path)
    model.set_classes(classes)
    print(f"Set classes ({len(classes)}): {classes}")

    # ================= 3. 初始化 Habitat 环境 =================
    scene_id = args.scene_id
    scene_path = f"data/scene_datasets/hm3d/val/{scene_id}/{scene_id.split('-')[-1]}.basis.glb"
    if not os.path.exists(scene_path):
        scene_path = os.path.abspath(scene_path)
    
    print(f"Loading Scene: {scene_path}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0

    robot = Robot()
    cfg = habitat_sim.Configuration(sim_cfg, [robot.agent_config])
    sim = habitat_sim.Simulator(cfg)
    
    class SimpleEnv:
        def __init__(self, simulator):
            self.sim = simulator
        def reset(self):
            self.sim.initialize_agent(0)
            if self.sim.pathfinder.is_loaded:
                agent = self.sim.get_agent(0)
                nav_point = self.sim.pathfinder.get_random_navigable_point()
                agent.set_state(habitat_sim.AgentState(position=nav_point))
            return self.sim.get_sensor_observations()
        def step(self, action):
            return self.sim.step(action)
        def close(self):
            self.sim.close()
            
    env = SimpleEnv(sim)
    
    # ================= 4. 初始化 Mapper (Probabilistic) =================
    mapper = HabitatMapper(map_size_meters=80.0, resolution=0.05, num_classes=len(classes))
    
    # ================= 5. 初始化控制器 =================
    controller = KeyboardController()

    # Reset
    obs = env.reset()
    mapper.reset()
    
    print("\n=== YOLO-World 主动探索模式 (概率融合版) ===")
    print("操作说明:")
    print("  W/A/S/D : 移动前后左右")
    print("  Q       : 退出并保存所有结果")
    
    # 窗口设置
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Semantic Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Entropy Map", cv2.WINDOW_NORMAL) # [NEW] 熵图窗口
    
    frames = []
    
    try:
        step_count = 0
        while True:
            # 1. 获取观测
            rgb_obs = obs["rgb_sensor"]
            depth_obs = obs["depth_sensor"]
            
            if rgb_obs.shape[2] == 4:
                rgb_img = rgb_obs[:, :, :3]
            else:
                rgb_img = rgb_obs
            
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            # 2. YOLO-World 推理
            results = model.predict(bgr_img, conf=args.conf, verbose=False)
            result = results[0]
            
            # 3. 构建伪语义图 和 置信度图
            H, W = depth_obs.shape
            semantic_obs = np.full((H, W), -1, dtype=np.int32)
            confidence_obs = np.zeros((H, W), dtype=np.float32) # [NEW]
            
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy().astype(float) # [NEW]
            
            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = box
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                semantic_obs[y1:y2, x1:x2] = cls_id
                confidence_obs[y1:y2, x1:x2] = conf # [NEW] 填充置信度

            # 4. 更新 Mapper (传入 confidence)
            agent_state = env.sim.get_agent(0).get_state()
            
            # 查找 'floor' 的 ID，用于回调 (如果存在)
            floor_id = -1
            if "floor" in classes:
                floor_id = classes.index("floor")
                
            mapper.update(depth_obs, semantic_obs, agent_state, 
                          check_is_floor_callback=lambda cid: cid == floor_id,
                          confidence_obs=confidence_obs)
            
            # 5. 可视化 & 抓帧
            # 左边: YOLO 结果 (BGR)
            annotated_frame = result.plot()
            
            # 中间: 语义地图 (BGR)
            vis_map = mapper.get_semantic_map_colored(draw_agent=True)
            scale = H / vis_map.shape[0]
            vis_map_resized = cv2.resize(vis_map, (int(vis_map.shape[1] * scale), H))
            
            # 右边: [NEW] 熵图 (BGR Heatmap)
            entropy_map_vis = mapper.get_entropy_map_colored()
            entropy_map_resized = cv2.resize(entropy_map_vis, (int(entropy_map_vis.shape[1] * scale), H))
            
            # 实时显示
            cv2.imshow("YOLO Detection", annotated_frame)
            cv2.imshow("Semantic Map", vis_map_resized)
            cv2.imshow("Entropy Map", entropy_map_resized)
            
            # 收集帧 (全景拼接)
            vis_yolo_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            vis_map_rgb = cv2.cvtColor(vis_map_resized, cv2.COLOR_BGR2RGB)
            vis_ent_rgb = cv2.cvtColor(entropy_map_resized, cv2.COLOR_BGR2RGB)
            
            combined = np.hstack((vis_yolo_rgb, vis_map_rgb, vis_ent_rgb))
            frames.append(combined)
            
            step_count += 1
            
            # 6. 控制逻辑
            key = cv2.waitKey(100) 
            
            if key != -1:
                action_name = controller.get_action(key)
                if action_name in ["quit", "save_and_quit"]:
                    print("\nQuitting...")
                    break
                elif action_name is not None:
                    obs = env.step(action_name)
                    print(f"Action: {action_name}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n[Error] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 无论正常退出还是报错，尝试保存结果
        print("\nSaving results...")
        
        try:
            # 1. Save GIF
            gif_path = os.path.join(output_dir, "exploration.gif")
            if len(frames) > 0:
                print(f"Saving GIF to {gif_path}...")
                imageio.mimsave(gif_path, frames, duration=0.2)
                
            # 2. Save Maps
            np.save(os.path.join(output_dir, "semantic_map.npy"), mapper.get_semantic_map_colored(draw_agent=False)) 
            np.save(os.path.join(output_dir, "semantic_counts.npy"), mapper.semantic_counts) 
            
            # 3. Save Entropy
            np.save(os.path.join(output_dir, "entropy_map.npy"), mapper.get_entropy_map_colored())
            
            print(f"All results saved to {output_dir}")
        except Exception as save_err:
            print(f"Error while saving results: {save_err}")

        env.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
