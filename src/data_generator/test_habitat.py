import habitat_sim
import habitat
import numpy as np

print(f"Habitat-Sim Version: {habitat_sim.__version__}")
print(f"Habitat-Lab Version: {habitat.__version__}")

# 测试一下 Quaternion (您的代码中用到)
import quaternion
q = np.quaternion(1, 0, 0, 0)
print(f"Quaternion test: {q}")

print("✅ 环境安装成功！")