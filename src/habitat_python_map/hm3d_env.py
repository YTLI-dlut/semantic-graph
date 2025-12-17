import os
import shutil
import habitat_sim

# ================= 路径配置 =================
ROOT_DIR = "/home/iiau/HM3D"
DIR_HABITAT = os.path.join(ROOT_DIR, "hm3d-val-habitat-v0.2")
DIR_SEMANTIC = os.path.join(ROOT_DIR, "hm3d-val-semantic-annots-v0.2")
DIR_CONFIGS = os.path.join(ROOT_DIR, "hm3d-val-semantic-configs-v0.2")
UNIFIED_DIR = os.path.join(ROOT_DIR, "hm3d_unified_val")
# ===========================================

class HM3DEnvironment:
    def __init__(self, scene_id, agent_config):
        """
        初始化环境
        :param scene_id: 场景 ID (e.g., "00800-TEEsavR23oF")
        :param agent_config: 从 Robot 类获取的 agent 配置
        """
        self.scene_id = scene_id
        # 1. 自动整理数据 (确保官方配置能找到文件)
        self.config_path = self._organize_dataset()
        
        # 2. 创建仿真器配置
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_dataset_config_file = self.config_path
        sim_cfg.enable_physics = True
        
        # 构造官方格式的场景句柄: "文件夹/文件名.basis.glb"
        target_handle = f"{scene_id}/{scene_id.split('-')[-1]}.basis.glb"
        sim_cfg.scene_id = target_handle
        print(f"--- [Env] Loading Scene: {target_handle} ---")

        # 3. 启动仿真器
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_config]))
        
        # 4. 初始化语义场景 (预热)
        _ = self.sim.semantic_scene

    def _organize_dataset(self):
        """ 自动创建软链接，重组数据集结构 """
        target_dir = os.path.join(UNIFIED_DIR, self.scene_id)
        os.makedirs(target_dir, exist_ok=True)
        
        # 需要链接的文件后缀
        suffixes = [".basis.glb", ".basis.navmesh", ".semantic.glb", ".semantic.txt"]
        # 来源目录映射
        sources = {
            ".basis.glb": DIR_HABITAT,
            ".basis.navmesh": DIR_HABITAT,
            ".semantic.glb": DIR_SEMANTIC,
            ".semantic.txt": DIR_SEMANTIC
        }

        for suffix in suffixes:
            src_dir = os.path.join(sources[suffix], self.scene_id)
            if not os.path.exists(src_dir): continue
            
            files = [f for f in os.listdir(src_dir) if f.endswith(suffix)]
            for f in files:
                src = os.path.join(src_dir, f)
                dst = os.path.join(target_dir, f)
                if not (os.path.exists(dst) or os.path.islink(dst)):
                    os.symlink(src, dst)

        # 复制配置文件
        cfg_name = "hm3d_annotated_val_basis.scene_dataset_config.json"
        src_cfg = os.path.join(DIR_CONFIGS, cfg_name)
        dst_cfg = os.path.join(UNIFIED_DIR, cfg_name)
        if not os.path.exists(dst_cfg):
            shutil.copy(src_cfg, dst_cfg)
            
        return dst_cfg

    def reset(self):
        # 随机位置初始化
        agent = self.sim.initialize_agent(0)
        if self.sim.pathfinder.is_loaded:
            agent.set_state(habitat_sim.AgentState(position=self.sim.pathfinder.get_random_navigable_point()))
        return self.sim.get_sensor_observations()

    def step(self, action_name):
        return self.sim.step(action_name)

    def close(self):
        self.sim.close()