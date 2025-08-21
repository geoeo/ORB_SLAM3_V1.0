#pragma once

#include<queue>
#include<mutex>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>



namespace ros2_orbslam3 {
    class ImuGrabber
    {
    public:
        ImuGrabber(rclcpp::Logger logger): logger_(logger){};
        void GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg);

        std::queue<sensor_msgs::msg::Imu> imuBuf;
        std::mutex mBufMutex;

    private:
        rclcpp::Logger logger_;
    };

    void ImuGrabber::GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
    {
    mBufMutex.lock();
    RCLCPP_INFO_STREAM(logger_,"IMU received - Buffer size: " << imuBuf.size());
    imuBuf.push(*imu_msg);
    mBufMutex.unlock();
    }
}

