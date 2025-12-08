#!/usr/bin/env python3
import rospy
import habitat_sim
import numpy as np
import math
import cv2
import os
import tf
import glob
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point, Twist
from std_msgs.msg import Float64

from tf.transformations import quaternion_from_euler

# === 1. ApexNav 专用旋转计算 (用于坐标系转换) ===
def apexnav_quaternion_from_euler(ai, aj, ak):
    ai /= 2.0; aj /= 2.0; ak /= 2.0
    ci = math.cos(ai); si = math.sin(ai)
    cj = math.cos(aj); sj = math.sin(aj)
    ck = math.cos(ak); sk = math.sin(ak)
    cc = ci*ck; cs = ci*sk; sc = si*ck; ss = si*sk
    q = np.empty((4, )); q[0] = cj*sc - sj*cs; q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc; q[3] = cj*cc + sj*ss
    return q

# === 2. 配置生成器 ===
def make_cfg(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True

    sensor_specs = []
    
    # 深度相机
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 0.88, 0.0]
    depth_spec.orientation = [0, 0.0, 0.0] # 下看 30 度
    sensor_specs.append(depth_spec)

    # RGB 相机
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 640]
    rgb_spec.position = [0.0, 0.88, 0.0]
    rgb_spec.orientation = [0, 0.0, 0.0]
    sensor_specs.append(rgb_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

class HabitatROSPublisher:
    def __init__(self):
        self.depth_pub = rospy.Publisher("/habitat/camera_depth", Image, queue_size=10)
        self.odom_pub = rospy.Publisher("/habitat/odom", Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher("/habitat/sensor_pose", Odometry, queue_size=10)
        self.conf_pub = rospy.Publisher("/detector/confidence_threshold", Float64, queue_size=10, latch=True)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback, queue_size=1)
        
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.next_action = None
        self.MAX_DEPTH = 10.0
        
        self.conf_pub.publish(0.5)
        print(f"ApexNav Bridge (Fixed Rotation). Norm Range: 0 - {self.MAX_DEPTH}m")

    def cmd_vel_callback(self, msg):
        if msg.linear.x > 0.1: self.next_action = "move_forward"
        elif msg.angular.z > 0.1: self.next_action = "turn_left"
        elif msg.angular.z < -0.1: self.next_action = "turn_right"

    def publish(self, observations, agent):
        ros_time = rospy.Time.now()
        agent_state = agent.get_state()
        
        # 1. 深度图发布
        depth_metric = observations["depth_sensor"]
        depth_metric = np.clip(depth_metric, 0, self.MAX_DEPTH)
        depth_normalized = depth_metric / self.MAX_DEPTH
        if depth_normalized.dtype != np.float32: 
            depth_normalized = depth_normalized.astype(np.float32)
            
        msg = Image(); msg.header.stamp = ros_time; msg.header.frame_id = "world"
        msg.height = depth_normalized.shape[0]; msg.width = depth_normalized.shape[1]
        msg.encoding = "32FC1"; msg.is_bigendian = 0; msg.step = msg.width * 4
        msg.data = depth_normalized.tobytes()
        self.depth_pub.publish(msg)

        # 2. 状态解算
        gps = agent_state.position
        s_state = agent_state.sensor_states["depth_sensor"]
        rot = s_state.rotation
        
        mat = tf.transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        fwd = np.dot(mat, [0, 0, -1, 0])
        
        real_pitch = math.asin(fwd[1])
        # 修正方向: atan2(-x, -z)
        real_compass = math.atan2(-fwd[0], -fwd[2])

        # 3. 坐标转换 & 发布
        s_pos = s_state.position
        ros_cam_pos = Point(-s_pos[2], -s_pos[0], s_pos[1])
        apex_quat = apexnav_quaternion_from_euler(real_pitch + np.pi/2.0, np.pi, real_compass + np.pi/2.0)
        
        ros_base_pos = Point(-gps[2], -gps[0], 0.0)
        base_quat = apexnav_quaternion_from_euler(0, 0, real_compass)

        sensor_pose = Odometry()
        sensor_pose.header.stamp = ros_time; sensor_pose.header.frame_id = "world"; sensor_pose.child_frame_id = "camera_link"
        sensor_pose.pose.pose.position = ros_cam_pos; sensor_pose.pose.pose.orientation = Quaternion(*apex_quat)
        self.pose_pub.publish(sensor_pose)

        odom = Odometry()
        odom.header.stamp = ros_time; odom.header.frame_id = "world"; odom.child_frame_id = "base_link"
        odom.pose.pose.position = ros_base_pos; odom.pose.pose.orientation = Quaternion(*base_quat)
        self.odom_pub.publish(odom)

        self.tf_broadcaster.sendTransform((ros_base_pos.x, ros_base_pos.y, ros_base_pos.z), base_quat, ros_time, "base_link", "world")
        self.tf_broadcaster.sendTransform((ros_cam_pos.x, ros_cam_pos.y, ros_cam_pos.z), apex_quat, ros_time, "camera_link", "world")
        self.conf_pub.publish(0.5)

def main():
    rospy.init_node('habitat_bridge')
    
    scene_root = os.path.expanduser("~/createGraph_ws/data/scene_datasets/hm3d/val")
    search_path = os.path.join(scene_root, "*/*.glb")
    scenes = glob.glob(search_path)
    valid_scenes = [s for s in scenes if "basis" not in s and "semantic" not in s]
    
    if not valid_scenes:
        print("Error: No scenes found in hm3d/val. Check data path.")
        return
        
    scene_path = valid_scenes[0]
    print(f"Loading: {os.path.basename(scene_path)}")

    try:
        cfg = make_cfg(scene_path)
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"Init Error: {e}")
        return

    pub = HabitatROSPublisher()
    
    agent = sim.get_agent(0)
    if sim.pathfinder.is_loaded:
        start_state = habitat_sim.AgentState()
        start_state.position = sim.pathfinder.get_random_navigable_point()
        # 这里调用的是 tf.transformations.quaternion_from_euler
        start_state.rotation = quaternion_from_euler(0, 0, 0)
        agent.set_state(start_state)
    
    print("\n[Running] Fixed Rotation Mode. Use teleop to control.\n")
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if pub.next_action:
            sim.step(pub.next_action)
            pub.next_action = None
        observations = sim.get_sensor_observations()
        pub.publish(observations, agent)
        rate.sleep()
    sim.close()

if __name__ == "__main__":
    main()