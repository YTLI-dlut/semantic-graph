#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class ApexNavMapper {
public:
    ApexNavMapper() : nh_("~") {
        nh_.param("fx", fx_, 256.0);
        nh_.param("max_dist", max_dist_, 5.0);
        nh_.param("min_height", min_height_, 0.5);
        nh_.param("max_height", max_height_, 1.5);
        nh_.param("resolution", resolution_, 0.05);

        depth_sub_ = nh_.subscribe("/habitat/camera_depth", 1, &ApexNavMapper::depthCb, this);
        pose_sub_ = nh_.subscribe("/habitat/sensor_pose", 1, &ApexNavMapper::poseCb, this);
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/map", 1);
        scan_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/habitat/scan_cloud", 1);

        map_width_ = 600;
        map_height_ = 600;
        grid_map_.resize(map_width_ * map_height_, 0);
    }

    void poseCb(const nav_msgs::Odometry::ConstPtr& msg) {
        // Eigen Quaternion 构造顺序是 (w, x, y, z)
        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                             msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        cam_pos_ << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
        cam_rot_ = q.toRotationMatrix();
        has_pose_ = true;

        // DEBUG: 计算并打印当前的偏航角 (Yaw)
        auto euler = cam_rot_.eulerAngles(0, 1, 2); // ZYX order roughly, but purely for debug
        // 简单的 Yaw 计算
        double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
        current_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    }

    void depthCb(const sensor_msgs::Image::ConstPtr& msg) {
        if (!has_pose_) return;

        double width = msg->width;
        double height = msg->height;
        fx_ = width / 2.0;
        fy_ = width / 2.0;
        cx_ = width / 2.0;
        cy_ = height / 2.0;

        cv_bridge::CvImagePtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(msg, "32FC1"); } catch (...) { return; }
        cv::Mat depth_img = cv_ptr->image;
        
        sensor_msgs::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = ros::Time::now();
        cloud_msg.header.frame_id = "world";
        cloud_msg.height = 1; cloud_msg.width = 0;
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(depth_img.rows * depth_img.cols);
        
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        
        int valid_pts = 0;
        int step = 2;

        // --- DEBUG 变量 ---
        // 我们只追踪图像正中心的一个像素点来调试
        int debug_u = depth_img.cols / 2;
        int debug_v = depth_img.rows / 2;
        bool debug_printed = false;

for (int v = 0; v < depth_img.rows; v += step) {
            for (int u = 0; u < depth_img.cols; u += step) {
                float d = depth_img.at<float>(v, u);
                
                if (d <= 0.1 || d > max_dist_) continue;

                // 1. 原始计算 (这是相机光学坐标系: Z=深度, X=右, Y=下)
                Eigen::Vector3d pt_optical;
                pt_optical(0) = (u - cx_) * d / fx_;
                pt_optical(1) = (v - cy_) * d / fy_;
                pt_optical(2) = d;

                // 2. 【关键修正】转换为机器人机身坐标系 (Body Frame)
                // 标准 ROS 转换关系：
                // Body X (前) <--- Optical Z (深)
                // Body Y (左) <--- Optical -X (左)
                // Body Z (上) <--- Optical -Y (上)
                Eigen::Vector3d pt_body;
                pt_body << pt_optical.z(), -pt_optical.x(), -pt_optical.y();

                // 3. 变换到世界坐标系
                // 现在 cam_rot_ 作用在正确的 Body 向量上了
                Eigen::Vector3d pt_world = cam_rot_ * pt_body + cam_pos_;

                // --- DEBUG (为了验证修复，你可以保留这个打印) ---
                if (!debug_printed && std::abs(u - debug_u) < step && std::abs(v - debug_v) < step) {
                     ROS_INFO_THROTTLE(1.0, 
                        "\n--- DEBUG INFO (FIXED) ---\n"
                        "Robot Yaw: %.2f\n"
                        "1. Optical: [%.2f, %.2f, %.2f] (Z=Depth)\n"
                        "2. Body:    [%.2f, %.2f, %.2f] (X=Forward)\n"
                        "3. World:   [%.2f, %.2f, %.2f]\n"
                        "--------------------", 
                        current_yaw_,
                        pt_optical.x(), pt_optical.y(), pt_optical.z(),
                        pt_body.x(), pt_body.y(), pt_body.z(),
                        pt_world.x(), pt_world.y(), pt_world.z()
                    );
                    debug_printed = true;
                }

                // 4. 地图更新 (后续逻辑不变)
                if (pt_world.z() > min_height_ && pt_world.z() < max_height_) {
                    updateMap(pt_world.x(), pt_world.y(), 100);
                    *iter_x = pt_world.x();
                    *iter_y = pt_world.y();
                    *iter_z = pt_world.z();
                    ++iter_x; ++iter_y; ++iter_z; valid_pts++;
                } 
                else if (pt_world.z() < min_height_) {
                    updateMap(pt_world.x(), pt_world.y(), 0);
                }
            }
        }
        modifier.resize(valid_pts);
        scan_pub_.publish(cloud_msg);
        publishMap();
    }

    void updateMap(double x, double y, int value) {
        int idx_x = (x / resolution_) + (map_width_ / 2);
        int idx_y = (y / resolution_) + (map_height_ / 2);
        if (idx_x >= 0 && idx_x < map_width_ && idx_y >= 0 && idx_y < map_height_) {
            if (value == 100) grid_map_[idx_y * map_width_ + idx_x] = 100;
            else if (grid_map_[idx_y * map_width_ + idx_x] != 100) grid_map_[idx_y * map_width_ + idx_x] = 0;
        }
    }

    void publishMap() {
        nav_msgs::OccupancyGrid map_msg;
        map_msg.header.stamp = ros::Time::now();
        map_msg.header.frame_id = "world";
        map_msg.info.resolution = resolution_;
        map_msg.info.width = map_width_;
        map_msg.info.height = map_height_;
        map_msg.info.origin.position.x = -(map_width_ * resolution_) / 2.0;
        map_msg.info.origin.position.y = -(map_height_ * resolution_) / 2.0;
        map_msg.data = grid_map_;
        map_pub_.publish(map_msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber depth_sub_, pose_sub_;
    ros::Publisher map_pub_, scan_pub_;
    double fx_, fy_, cx_, cy_, max_dist_, min_height_, max_height_, resolution_;
    Eigen::Vector3d cam_pos_;
    Eigen::Matrix3d cam_rot_;
    bool has_pose_ = false;
    double current_yaw_ = 0.0; // Debug 用
    std::vector<int8_t> grid_map_;
    int map_width_, map_height_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "simple_mapper");
    ApexNavMapper mapper;
    ros::spin();
    return 0;
}