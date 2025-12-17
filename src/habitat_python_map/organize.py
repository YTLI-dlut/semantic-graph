import os
import shutil

# ================= 配置区域 =================
# 你的 HM3D 原始数据根目录
ROOT_DIR = "/home/iiau/HM3D"

# 源文件夹名称 (根据你提供的 tree 结构)
DIR_GLB = os.path.join(ROOT_DIR, "hm3d-val-glb-v0.2") # 视觉 GLB
DIR_HABITAT = os.path.join(ROOT_DIR, "hm3d-val-habitat-v0.2") # 物理/导航数据
DIR_SEMANTIC = os.path.join(ROOT_DIR, "hm3d-val-semantic-annots-v0.2") # 语义数据
DIR_CONFIGS = os.path.join(ROOT_DIR, "hm3d-val-semantic-configs-v0.2") # 配置文件

# 目标统一文件夹 (生成在这里)
UNIFIED_DIR = os.path.join(ROOT_DIR, "hm3d_unified_val")

# 目标场景 ID
SCENE_ID = "00800-TEEsavR23oF"
# ===========================================

def organize():
    print(f"--- 正在构建统一数据集目录: {UNIFIED_DIR} ---")
    
    # 1. 创建场景子目录
    target_scene_dir = os.path.join(UNIFIED_DIR, SCENE_ID)
    os.makedirs(target_scene_dir, exist_ok=True)
    
    # 2. 定义任务列表: (源目录, 后缀)
    # 注意：根据官方 config，它寻找的是 *.basis.glb, *.basis.navmesh, *.semantic.glb, *.semantic.txt
    tasks = [
        (os.path.join(DIR_HABITAT, SCENE_ID), ".basis.glb"),
        (os.path.join(DIR_HABITAT, SCENE_ID), ".basis.navmesh"),
        (os.path.join(DIR_SEMANTIC, SCENE_ID), ".semantic.glb"),
        (os.path.join(DIR_SEMANTIC, SCENE_ID), ".semantic.txt"),
    ]
    
    # 3. 创建软链接
    for src_dir, suffix in tasks:
        if not os.path.exists(src_dir):
            print(f"[跳过] 目录不存在: {src_dir}")
            continue
            
        files = [f for f in os.listdir(src_dir) if f.endswith(suffix)]
        for f in files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(target_scene_dir, f)
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            print(f"-> 链接成功: {f}")

    # 4. 复制配置文件
    cfg_name = "hm3d_annotated_val_basis.scene_dataset_config.json"
    src_cfg = os.path.join(DIR_CONFIGS, cfg_name)
    dst_cfg = os.path.join(UNIFIED_DIR, cfg_name)
    
    if os.path.exists(dst_cfg):
        os.remove(dst_cfg)
    shutil.copy(src_cfg, dst_cfg)
    print(f"-> 配置文件已就位: {cfg_name}")
    print("\n数据整理完成！现在可以使用 control.py 加载了。")

if __name__ == "__main__":
    organize()