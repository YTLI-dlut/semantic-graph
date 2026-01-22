from ultralytics import YOLOWorld
import cv2
import time

# 1. 加载模型
# 'yolov8s-world.pt' 是轻量版，速度快，适合实时 RL
# 'yolov8l-world.pt' 精度更高，显存占用稍大
print("正在加载模型...")
model = YOLOWorld('yolov8s-world.pt')  

# 2. 定义你的自定义词汇表 (室内导航常用)
# 这就是 YOLO-World 的核心：无需训练，直接告诉它找什么
classes = ["chair", "table", "sofa", "bed", "tv", "person", "bus", "window", "wall"]

# 3. 设置类别 (离线加速模式)
# 这一步会将文本特征编码并固定，极大提升后续推理速度
model.set_classes(classes)

# 4. 读取一张图片 (或者使用摄像头/仿真器截图)
# 这里我们用网上的一张图做测试
img_url = "https://ultralytics.com/images/bus.jpg"

# 5. 推理
print("开始推理...")
start_time = time.time()
results = model.predict(img_url)
end_time = time.time()

print(f"推理完成，耗时: {(end_time - start_time)*1000:.2f} ms")

# 6. 显示结果
# results[0].show() # 如果在有桌面的 Ubuntu 上运行
# 或者保存图片查看
results[0].save("output.jpg")
print("结果已保存为 output.jpg")