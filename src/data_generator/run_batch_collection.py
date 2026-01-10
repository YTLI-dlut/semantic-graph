
import os
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_file", type=str, default="data/valid_scenes.txt", help="Path to valid scenes list")
    parser.add_argument("--samples", type=int, default=2000, help="Number of teleport samples per floor")
    args = parser.parse_args()

    if not os.path.exists(args.scenes_file):
        print(f"Error: Scenes file not found: {args.scenes_file}")
        return

    # 读取场景列表
    with open(args.scenes_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(scenes)} scenes from {args.scenes_file}")
    
    total_tasks = len(scenes) * 2 # Floor 0 and 1
    completed_tasks = 0
    
    start_all = time.time()

    for idx, scene_id in enumerate(scenes):
        print(f"\n[{idx+1}/{len(scenes)}] Processing Scene: {scene_id}")
        
        for floor_idx in [0, 1]:
            print(f"  -> Starting Floor {floor_idx}...")
            
            # 构造命令
            cmd = [
                "python", "src/data_generator/run_auto_floor_mapper.py",
                "--scene_id", scene_id,
                "--floor", str(floor_idx),
                "--samples", str(args.samples)
            ]
            
            # 执行
            try:
                start_task = time.time()
                # 使用 check=True 能够捕获非零退出码（如果脚本崩了）
                # 使用 capture_output=False 让子进程输出直接打印到屏幕，方便观察进度
                subprocess.run(cmd, check=False) 
                
                elapsed = time.time() - start_task
                print(f"  -> Floor {floor_idx} finished in {elapsed:.1f}s.")
                
            except Exception as e:
                print(f"  [Error] Failed to process {scene_id} Floor {floor_idx}: {e}")
                
            completed_tasks += 1
            print(f"  Progress: {completed_tasks}/{total_tasks} tasks complated.")

    total_elapsed = time.time() - start_all
    print(f"\nBatch collection complete in {total_elapsed:.1f}s.")

if __name__ == "__main__":
    main()
