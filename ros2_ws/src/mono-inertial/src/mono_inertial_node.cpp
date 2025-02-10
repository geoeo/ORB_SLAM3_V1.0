#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>
#include<tuple>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <CUDACvManagedMemory/cuda_cv_managed_memory.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "cuda_runtime.h"
#include <ORB_SLAM3/System.h>
#include <ORB_SLAM3/ImuTypes.h>

using namespace std::chrono_literals;
using namespace cuda_cv_managed_memory;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg);

    queue<sensor_msgs::msg::Imu> imuBuf;
    std::mutex mBufMutex;
};

void ImuGrabber::GrabImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(*imu_msg);
  mBufMutex.unlock();
  return;
}

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe, double tshift_cam_imu, float resize_factor, const cv::cuda::GpuMat &undistortion_map_1, const cv::cuda::GpuMat& undistortion_map_2, const cv::cuda::GpuMat& undistorted_image_gpu, const cv::cuda::GpuMat& resized_img_gpu)
      : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), timeshift_cam_imu(tshift_cam_imu),count(0), img_resize_factor(resize_factor),
        m_undistortion_map_1(undistortion_map_1), m_undistortion_map_2(undistortion_map_2), m_undistorted_image_gpu(undistorted_image_gpu), m_stream(cv::cuda::Stream()){

            auto new_rows = static_cast<int>(undistorted_image_gpu.rows*img_resize_factor);
            auto new_cols = static_cast<int>(undistorted_image_gpu.cols*img_resize_factor);

            m_resized_img_gpu = std::shared_ptr<CUDAManagedMemory>(new CUDAManagedMemory(new_rows*new_cols*3, new_rows, new_cols, CV_8UC3, new_cols*3),CUDAManagedMemoryDeleter());

            
      }

    void GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    CUDAManagedMemory::SharedPtr GetImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
    void SyncWithImu();

    queue<sensor_msgs::msg::Image::ConstSharedPtr> img0Buf;
    std::mutex mBufMutex;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(4, 4));
    double timeshift_cam_imu;
    uint64_t count;
    float img_resize_factor;

    cv::cuda::GpuMat m_undistortion_map_1;
    cv::cuda::GpuMat m_undistortion_map_2;
    cv::cuda::GpuMat m_undistorted_image_gpu;
    CUDAManagedMemory::SharedPtr m_resized_img_gpu; 
    cv::cuda::Stream m_stream;

};

void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  mBufMutex.lock();
  img0Buf.push(img_msg);
  count = 0;
  

  mBufMutex.unlock();
}

CUDAManagedMemory::SharedPtr ImageGrabber::GetImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  auto width = img_msg->width;
  auto height = img_msg->height;
  auto size_in_bytes = img_msg->height*img_msg->step;

  auto cuda_managed_memory_image 
    = std::shared_ptr<CUDAManagedMemory>(new CUDAManagedMemory(size_in_bytes, height, width, CV_8UC3, img_msg->step),CUDAManagedMemoryDeleter());
  if(cudaMemcpy(cuda_managed_memory_image->getRaw(), &img_msg->data[0], size_in_bytes, cudaMemcpyDefault) != cudaError_t::cudaSuccess)
        throw std::runtime_error("CUDAManagedMemory - Failed to copy memory to CUDA unified");

  cv::cuda::remap(cuda_managed_memory_image->getCvGpuMat(), m_undistorted_image_gpu, m_undistortion_map_1, m_undistortion_map_2, cv::InterpolationFlags::INTER_CUBIC,cv::BORDER_CONSTANT,cv::Scalar(),m_stream);
  auto new_rows = static_cast<int>(height*img_resize_factor);
  auto new_cols = static_cast<int>(width*img_resize_factor);
  cv::Size new_im_size = cv::Size(new_cols,new_rows);
  m_stream.waitForCompletion();
  cv::cuda::resize(m_undistorted_image_gpu, m_resized_img_gpu->getCvGpuMat(), new_im_size, 0, 0, cv::INTER_LINEAR, m_stream);

  CUDAManagedMemory::SharedPtr cuda_managed_memory_image_grey = std::shared_ptr<CUDAManagedMemory>(new CUDAManagedMemory(new_rows*new_cols, new_rows, new_cols, CV_8UC1, new_cols),CUDAManagedMemoryDeleter());
  cv::cvtColor(m_resized_img_gpu->getCvMat(),cuda_managed_memory_image_grey->getCvMat(),cv::COLOR_BGR2GRAY);

  return cuda_managed_memory_image_grey;
}


void ImageGrabber::SyncWithImu()
{
  double init_ts = 0;
  while(1)
  {
    CUDAManagedMemory::SharedPtr im_managed;
    cv::Mat im;
    double tIm = 0;
    if (!img0Buf.empty()&&!mpImuGb->imuBuf.empty())
    {
      mpImuGb->mBufMutex.lock();
      auto ros_imu_ts = rclcpp::Time(mpImuGb->imuBuf.front().header.stamp);
      if(init_ts == 0)
        init_ts = ros_imu_ts.seconds();
      mpImuGb->mBufMutex.unlock();

      this->mBufMutex.lock();
      im_managed = GetImage(img0Buf.front());
      auto ros_image_ts_front =  rclcpp::Time(img0Buf.front()->header.stamp);
      tIm = ros_image_ts_front.seconds() + timeshift_cam_imu - init_ts;
      img0Buf.pop();
      this->mBufMutex.unlock();
      
      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        auto imu_meas = mpImuGb->imuBuf.front();
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

          mpImuGb->imuBuf.pop();
          if(mpImuGb->imuBuf.empty())
            break;
          imu_meas = mpImuGb->imuBuf.front();
          ros_imu_ts_front = rclcpp::Time(imu_meas.header.stamp);
          t = ros_imu_ts_front.seconds();
          t-=init_ts;
        }
      }
      mpImuGb->mBufMutex.unlock();

      if(!vImuMeas.empty() && init_ts != 0){
        std::cout << "IMU meas size: " << vImuMeas.size() << std::endl;
        auto tracking_results = mpSLAM->TrackMonocular(im_managed,tIm,vImuMeas);
        Sophus::Matrix4f pose = std::get<0>(tracking_results).matrix();
        bool ba_complete_for_frame = std::get<1>(tracking_results);
        auto scale_factors = std::get<2>(tracking_results);
        vImuMeas.clear();
        cout << "BA completed: " << ba_complete_for_frame << endl;
        if(!scale_factors.empty())
          cout << "Latest Scale Factor: " << scale_factors.back() << endl;
        cout << "Current ts: " << tIm << endl;
        for(auto s : scale_factors)
          cout << " scale: " << s;
        cout << endl;
        //cout << pose(0,0) << ", " << pose(0,1) << ", " << pose(0,2) << ", " << pose(0,3) << endl;
        //cout << pose(1,0) << ", " << pose(1,1) << ", " << pose(1,2) << ", " << pose(1,3) << endl;
        //cout << pose(2,0) << ", " << pose(2,1) << ", " << pose(2,2) << ", " << pose(2,3) << endl;
        //cout << pose(3,0) << ", " << pose(3,1) << ", " << pose(3,2) << ", " << pose(3,3) << endl;
        cout << pose(0,3) << ", " << pose(1,3) << ", " << pose(2,3) << endl;
      }

    }

    //std::chrono::milliseconds tSleep(1);
    //std::this_thread::sleep_for(tSleep);
  }
}





/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class SlamNode : public rclcpp::Node
{
  public:
    SlamNode(std::string path_to_vocab, bool bEqual) : Node("mono_intertial_node"), path_to_vocab_(path_to_vocab), bEqual_(bEqual)
    {
      float resize_factor = 0.6;

      // F4
      ORB_SLAM3::CameraParameters cam{};
      cam.K = cv::Mat::zeros(3,3,CV_32F);
      cam.K.at<float>(0,0) = 1336.9226074186074;
      cam.K.at<float>(1,1) = 1335.4255026142991;
      cam.K.at<float>(0,2) = 1020.6417436861941;
      cam.K.at<float>(1,2) = 787.2614408235355;
      cam.K.at<float>(2,2) = 1;

      cv::Mat distCoeffs  = cv::Mat:: zeros(4,1,CV_32F);
      distCoeffs.at<float>(0,0) = -0.008025606047653894;
      distCoeffs.at<float>(1,0) = 0.06147505062442819;
      distCoeffs.at<float>(2,0) = -0.07203040867973515;
      distCoeffs.at<float>(3,0) = 0.035581570054134905;

      cam.distCoeffs = cv::Mat:: zeros(1,1,CV_32F); // We dont want to undistort im ORBSLAM so we pass dummy matrix

      cv::Mat m_undistortion_map1;
      cv::Mat m_undistortion_map2;

      cv::cuda::GpuMat m_undistortion_map_1;
      cv::cuda::GpuMat m_undistortion_map_2;
      cv::cuda::GpuMat m_undistorted_image_gpu = cv::cuda::GpuMat(1536, 2048, CV_8UC3);
      cv::cuda::GpuMat m_resized_img_gpu = cv::cuda::GpuMat(static_cast<int>(1536*resize_factor), static_cast<int>(2048*resize_factor), CV_8UC3);

      cv::fisheye::initUndistortRectifyMap(cam.K,
                        distCoeffs,
                        cv::Mat_<double>::eye(3, 3),
                        cam.K,
                        cv::Size(2048, 1536),
                        CV_32F,
                        m_undistortion_map1,
                        m_undistortion_map2);

      m_undistortion_map_1.upload(m_undistortion_map1);
      m_undistortion_map_2.upload(m_undistortion_map2);

      cam.K.at<float>(0,0) *= resize_factor;
      cam.K.at<float>(1,1) *= resize_factor;
      cam.K.at<float>(0,2) *= resize_factor;
      cam.K.at<float>(1,2) *= resize_factor;


      cam.fps        = 25;
      cam.orig_width      = static_cast<int>(2048*resize_factor);
      cam.orig_height     = static_cast<int>(1536*resize_factor);

      //1.0
      cam.new_width      = static_cast<int>(2048*resize_factor);
      cam.new_height     = static_cast<int>(1536*resize_factor);
      cam.isRGB      = false; // BGR

      ORB_SLAM3::OrbParameters orb{};
      orb.nFeatures   = 4000;
      orb.nLevels     = 4;
      orb.scaleFactor = 1.2;
      orb.minThFast   = 5;
      orb.iniThFast   = 15;
      orb.gridCount = 64;

      ORB_SLAM3::ImuParameters m_imu;
      m_imu.accelWalk  = 0.0006431373218006597; // x10
      m_imu.gyroWalk   = 0.000018714270991865037; //x10
      m_imu.noiseAccel =  0.015592957173554883; // x10
      m_imu.noiseGyro  = 0.0017019559710036963; // x10

      m_imu.InsertKFsWhenLost = false;

      cv::Mat cv_Tbc = cv::Mat::zeros(4,4,CV_32F);

      cv_Tbc.at<float>(0,0) =   0.01145462;
      cv_Tbc.at<float>(0,1) =   0.00408801;
      cv_Tbc.at<float>(0,2) =   0.99992604;
      cv_Tbc.at<float>(0,3) =   0.05702731;

      cv_Tbc.at<float>(1,0) =   0.00232179;
      cv_Tbc.at<float>(1,1) =   -0.99998906;
      cv_Tbc.at<float>(1,2) =   0.00406167;
      cv_Tbc.at<float>(1,3) =   -0.00619829;

      cv_Tbc.at<float>(2,0) =   0.9999317;
      cv_Tbc.at<float>(2,1) =   0.00227509;
      cv_Tbc.at<float>(2,2) =   -0.01146399;
      cv_Tbc.at<float>(2,3) =   -0.01762376;

      cv_Tbc.at<float>(3,0) =   0.0;
      cv_Tbc.at<float>(3,1) =   0.0;
      cv_Tbc.at<float>(3,2) =   0.0;
      cv_Tbc.at<float>(3,3) =   1.0;

      m_imu.Tbc = cv_Tbc;
      m_imu.freq = 400.0;

      double timeshift_cam_imu = 0.00851880502751802; // F4

      const int frame_grid_cols = 64;
      const int frame_grid_rows = 48;

      // Create SLAM system. It initializes all system threads and gets ready to process frames.
      SLAM_ = std::make_unique<ORB_SLAM3::System>(path_to_vocab_,cam, m_imu, orb, ORB_SLAM3::System::IMU_MONOCULAR, frame_grid_cols,frame_grid_rows,false, false);
      cout << "SLAM Init" << endl;

      igb_ = std::make_unique<ImageGrabber>(SLAM_.get(),&imugb_,bEqual_, timeshift_cam_imu, resize_factor, m_undistortion_map_1, m_undistortion_map_2, m_undistorted_image_gpu, m_resized_img_gpu);
      sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>("/bmi088_F4/imu", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImuGrabber::GrabImu, &imugb_, placeholders::_1));
      sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>("/AIT_Fighter4/down/image", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImageGrabber::GrabImage, igb_.get(), placeholders::_1));
      sync_thread_ = std::make_unique<std::thread>(&ImageGrabber::SyncWithImu,igb_.get());
    }

    ~SlamNode(){
      cout << "Trigger Shutdown" << endl;
      SLAM_->Shutdown();
    }

  private:
    std::string path_to_vocab_;
    bool bEqual_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_;
    ImuGrabber imugb_;
    std::unique_ptr<ImageGrabber> igb_;
    std::unique_ptr<std::thread> sync_thread_;
    std::unique_ptr<ORB_SLAM3::System> SLAM_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  bool bEqual = false;
  if(argc < 2 || argc > 3)
  {
    for(auto i = 0; i < argc; ++i)
      cout << argv[i] << endl;
    cerr  << endl << "Arg count: " << argc << endl << "Usage: ros2 run mono_inertial_node path_to_vocabulary [do_equalize]" << endl;
    rclcpp::shutdown();
    return 1;
  }


  if(argc==3)
  {
    std::string sbEqual(argv[2]);
    if(sbEqual == "true")
      bEqual = true;
  }


  rclcpp::spin(std::make_shared<SlamNode>(argv[1],bEqual));
  rclcpp::shutdown();
  return 0;
}