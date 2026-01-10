
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--habitat_dir", type=str, default="/home/iiau/HM3D/hm3d-val-habitat-v0.2", help="Path to HM3D habitat dataset")
    parser.add_argument("--semantic_dir", type=str, default="/home/iiau/HM3D/hm3d-val-semantic-annots-v0.2", help="Path to HM3D semantic annotations")
    parser.add_argument("--output_file", type=str, default="data/valid_scenes.txt", help="Output file path")
    args = parser.parse_args()

    if not os.path.exists(args.habitat_dir):
        print(f"Error: Habitat directory not found: {args.habitat_dir}")
        return
    
    if not os.path.exists(args.semantic_dir):
        print(f"Error: Semantic directory not found: {args.semantic_dir}")
        return

    print("Scanning scenes...")
    
    # 获取所有场景 ID (文件夹名)
    all_scenes = [d for d in os.listdir(args.habitat_dir) if os.path.isdir(os.path.join(args.habitat_dir, d))]
    all_scenes.sort()
    
    valid_scenes = []
    
    for scene_id in all_scenes:
        # 检查是否存在对应的语义文件夹
        sem_scene_dir = os.path.join(args.semantic_dir, scene_id)
        if not os.path.isdir(sem_scene_dir):
            continue
            
        # 检查是否存在 .semantic.txt 文件
        # HM3D 语义文件命名规则: {hash}.semantic.txt
        # scene_id 格式: {id}-{hash}
        scene_hash = scene_id.split('-')[-1]
        sem_file = os.path.join(sem_scene_dir, f"{scene_hash}.semantic.txt")
        
        if os.path.exists(sem_file):
            valid_scenes.append(scene_id)
            
    print(f"Total scenes scanned: {len(all_scenes)}")
    print(f"Valid scenes with semantics: {len(valid_scenes)}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        for s in valid_scenes:
            f.write(s + "\n")
            
    print(f"Valid scene list saved to: {args.output_file}")

if __name__ == "__main__":
    main()
