import cv2

class KeyboardController:
    def __init__(self):
        self.action_mapping = {
            ord('w'): "move_forward",
            ord('s'): "move_backward",
            ord('a'): "turn_left",
            ord('d'): "turn_right",
            ord('p'): "save_and_quit", # P 键保存并退出
            ord('q'): "save_and_quit", # Q 键也可以
            ord('j'): "jump", # J 键跳跃/瞬移
            27: "quit"  # ESC 键只退出不保存
        }

    def get_action(self, key):
        """
        根据按键返回动作名称
        :param key: cv2.waitKey 返回的按键值
        :return: 动作名称字符串 或 None
        """
        # 转换为小写，兼容大小写锁定
        if key == -1:
            return None
            
        # 处理一下 key 的掩码，防止不同系统差异
        key = key & 0xFF
        
        if key in self.action_mapping:
            return self.action_mapping[key]
            
        return None