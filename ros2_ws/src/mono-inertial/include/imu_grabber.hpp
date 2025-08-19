#pragma once

#include<queue>
#include<mutex>
#include <sensor_msgs/msg/imu.hpp>



namespace ros2_orbslam3 {
    class ImuGrabber
    {
    public:
        ImuGrabber(){};
        void GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg);

        std::queue<sensor_msgs::msg::Imu> imuBuf;
        std::mutex mBufMutex;
    };

    void ImuGrabber::GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
    {
    mBufMutex.lock();
    imuBuf.push(*imu_msg);
    mBufMutex.unlock();
    }
}

