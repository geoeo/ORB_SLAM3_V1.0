#pragma once

#include<vector>
#include<queue>
#include<mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/core.hpp>

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <imu_grabber.hpp>
#include <cuda_runtime.h>
#include <ORB_SLAM3/System.h>
#include <ORB_SLAM3/ImuTypes.h>


namespace ros2_orbslam3 {
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::NavSatFix> approximate_policy;
    class ImageGrabber
    {
    public:
        ImageGrabber(ORB_SLAM3::System* pSLAM, std::shared_ptr<ImuGrabber> pImuGb, const bool bClahe, double tshift_cam_imu, int width, int height, float resize_factor, double clahe_clip_limit, int clahe_grid_size,
        const cv::cuda::GpuMat &undistortion_map_1, const cv::cuda::GpuMat& undistortion_map_2, const cv::cuda::GpuMat& undistorted_image_gpu, rclcpp::Logger logger)
        : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), timeshift_cam_imu(tshift_cam_imu), img_resize_factor(resize_factor), m_width(width), m_height(height), logger_(logger),
            m_undistortion_map_1(undistortion_map_1), m_undistortion_map_2(undistortion_map_2), m_undistorted_image_gpu(undistorted_image_gpu), m_stream(cv::cuda::Stream())

        {
            m_resized_img_gpu = cv::cuda::HostMem(m_height, m_width, CV_8UC3, cv::cuda::HostMem::AllocType::SHARED);
            m_cuda_managed_memory_image_grey = cv::cuda::HostMem(m_height, m_width, CV_8UC1,cv::cuda::HostMem::AllocType::SHARED);
            m_cuda_managed_memory_image_grey_eq = cv::cuda::HostMem(m_height, m_width, CV_8UC1,cv::cuda::HostMem::AllocType::SHARED);
            mClahe = cv::cuda::createCLAHE(clahe_clip_limit, cv::Size(clahe_grid_size, clahe_grid_size));
        }

        void GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
        void GrabImageAndGNSS(const sensor_msgs::msg::Image::ConstSharedPtr img_msg, const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg_gnss);
        cv::cuda::HostMem ConvertImageToGPU(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
        void SyncWithImu();

        std::queue<sensor_msgs::msg::Image::ConstSharedPtr> img0Buf;
        std::mutex mBufMutex;
    
        ORB_SLAM3::System* mpSLAM;
        std::shared_ptr<ImuGrabber> mpImuGb;

        const bool mbClahe;
        double timeshift_cam_imu;
        float img_resize_factor;
        int m_width;
        int m_height;
        rclcpp::Logger logger_;

        cv::cuda::GpuMat m_undistortion_map_1;
        cv::cuda::GpuMat m_undistortion_map_2;
        cv::cuda::GpuMat m_undistorted_image_gpu;
        cv::cuda::HostMem m_resized_img_gpu; 
        cv::cuda::HostMem m_cuda_managed_memory_image_grey;
        cv::cuda::HostMem m_cuda_managed_memory_image_grey_eq;
        cv::cuda::Stream m_stream;
        cv::Ptr<cv::cuda::CLAHE> mClahe;

    };

    void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
    {
        mBufMutex.lock();
        //RCLCPP_INFO_STREAM(logger_,"Img received - Buffer size: " << img0Buf.size());
        img0Buf.push(img_msg);
        mBufMutex.unlock();
    }

    void ImageGrabber::GrabImageAndGNSS(const sensor_msgs::msg::Image::ConstSharedPtr img_msg, const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg_gnss)
    {
        RCLCPP_INFO_STREAM(logger_,"Img and GNSS received");
        //TODO
    }

    cv::cuda::HostMem ImageGrabber::ConvertImageToGPU(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
    {

        cv::Mat cv_im = cv_bridge::toCvShare(img_msg)->image;
        cv::cuda::HostMem cuda_managed_memory_image 
            = cv::cuda::HostMem(cv_im, cv::cuda::HostMem::AllocType::SHARED);
        cv::Size new_im_size = cv::Size(m_width,m_height);
        cv::cuda::remap(cuda_managed_memory_image.createGpuMatHeader(), m_undistorted_image_gpu, m_undistortion_map_1, m_undistortion_map_2, cv::InterpolationFlags::INTER_CUBIC,cv::BORDER_CONSTANT,cv::Scalar(),m_stream);

        cv::cuda::resize(m_undistorted_image_gpu, m_resized_img_gpu.createGpuMatHeader(), new_im_size, 0, 0, cv::INTER_LINEAR, m_stream);
        m_stream.waitForCompletion();
        cv::cuda::cvtColor(m_resized_img_gpu.createGpuMatHeader(),m_cuda_managed_memory_image_grey.createGpuMatHeader(),cv::COLOR_BGR2GRAY);
        mClahe->apply(m_cuda_managed_memory_image_grey.createGpuMatHeader(), m_cuda_managed_memory_image_grey_eq.createGpuMatHeader());
        return m_cuda_managed_memory_image_grey_eq;
    }


    void ImageGrabber::SyncWithImu()
    {
        double init_ts = 0;
        while(true)
        {
            cv::cuda::HostMem im_managed;
            cv::Mat im;
            double tIm = 0;
            if (!img0Buf.empty()&&!mpImuGb->imuBuf.empty())
            {
                mpImuGb->mBufMutex.lock();
                auto imu_front = mpImuGb->imuBuf.front();
                mpImuGb->mBufMutex.unlock();
                auto ros_imu_ts = rclcpp::Time(imu_front.header.stamp);
                if(init_ts == 0)
                    init_ts = ros_imu_ts.seconds();

                this->mBufMutex.lock();
                auto im_front = img0Buf.front();
                this->mBufMutex.unlock();
                im_managed = ConvertImageToGPU(im_front);
                auto ros_image_ts_front =  rclcpp::Time(im_front->header.stamp);
                tIm = ros_image_ts_front.seconds() + timeshift_cam_imu - init_ts;
                RCLCPP_INFO_STREAM(logger_,"Img buffer size: " << img0Buf.size());
                img0Buf.pop();

                
                vector<ORB_SLAM3::IMU::Point> vImuMeas;
                mpImuGb->mBufMutex.lock();
                auto imu_empty = mpImuGb->imuBuf.empty();
                mpImuGb->mBufMutex.unlock();
                if(!imu_empty)
                {
                    mpImuGb->mBufMutex.lock();
                    auto imu_meas = mpImuGb->imuBuf.front();
                    mpImuGb->mBufMutex.unlock();
                    auto ros_imu_ts_front = rclcpp::Time(imu_meas.header.stamp);
                    auto t = ros_imu_ts_front.seconds();
                    t-=init_ts;
                    // Load imu measurements from buffer
                    vImuMeas.clear();
                    while(t<=tIm)
                    { 
                        cv::Point3f acc(imu_meas.linear_acceleration.x, imu_meas.linear_acceleration.y, imu_meas.linear_acceleration.z);
                        cv::Point3f gyr(imu_meas.angular_velocity.x, imu_meas.angular_velocity.y, imu_meas.angular_velocity.z);
                        vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));

                        mpImuGb->mBufMutex.lock();
                        mpImuGb->imuBuf.pop();
                        if(mpImuGb->imuBuf.empty())
                            break;
                        imu_meas = mpImuGb->imuBuf.front();
                        mpImuGb->mBufMutex.unlock();
                        ros_imu_ts_front = rclcpp::Time(imu_meas.header.stamp);
                        t = ros_imu_ts_front.seconds();
                        t-=init_ts;
                    }
                }

                if(!vImuMeas.empty() && init_ts != 0){
                    RCLCPP_INFO_STREAM(logger_, "IMU meas size: " << vImuMeas.size());

                    while(!mpSLAM->getGlobalDataMutex()->try_lock())
                        this_thread::sleep_for(chrono::microseconds(500));

                    auto tracking_results = mpSLAM->TrackMonocular(im_managed,tIm,vImuMeas);
                    mpSLAM->getGlobalDataMutex()->unlock();
                    Sophus::Matrix4f pose = std::get<0>(tracking_results).matrix();
                    bool ba_complete_for_frame = std::get<1>(tracking_results);
                    bool is_keyframe = std::get<2>(tracking_results);
                    unsigned long int id = std::get<3>(tracking_results);
                    auto scale_factors = std::get<4>(tracking_results);
                    vImuMeas.clear();
                    RCLCPP_INFO_STREAM(logger_, "BA completed: " << ba_complete_for_frame);
                    if(!scale_factors.empty())
                        RCLCPP_INFO_STREAM(logger_, "Latest Scale Factor: " << scale_factors.back());
                    RCLCPP_INFO_STREAM(logger_, "Current ts: " << tIm);
                    for(auto s : scale_factors)
                        RCLCPP_INFO_STREAM(logger_, "Scale Factor: " << s);
                    RCLCPP_INFO_STREAM(logger_,"is keyframe: " << is_keyframe);
                    RCLCPP_INFO_STREAM(logger_, "pose: " << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3));
                }
            }
        }
    }
}