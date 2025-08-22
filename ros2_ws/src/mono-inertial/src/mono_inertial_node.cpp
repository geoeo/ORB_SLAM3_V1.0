#include<iostream>
#include<thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>


#include <imu_grabber.hpp>
#include <image_grabber.hpp>
#include <ORB_SLAM3/System.h>

using namespace ros2_orbslam3;

class SlamNode : public rclcpp::Node
{
  public:
    SlamNode(std::string path_to_vocab, bool bEqual) : Node("mono_intertial_node"), path_to_vocab_(path_to_vocab), bEqual_(bEqual)
    {
      float resize_factor = 0.3;

      // F6
      ORB_SLAM3::CameraParameters cam{};
      cam.K = cv::Mat::zeros(3,3,CV_32F);
      cam.K.at<float>(0,0) = 1340.8951163881811;
      cam.K.at<float>(1,1) = 1339.4354628196174;
      cam.K.at<float>(0,2) = 1007.6078987259765;
      cam.K.at<float>(1,2) = 767.9508975018316;
      cam.K.at<float>(2,2) = 1;

      cv::Mat distCoeffs  = cv::Mat:: zeros(4,1,CV_32F);
      distCoeffs.at<float>(0,0) = -0.0032741297195706454;
      distCoeffs.at<float>(1,0) = 0.044026087700189453;
      distCoeffs.at<float>(2,0) = -0.048481762487143865;
      distCoeffs.at<float>(3,0) = 0.0241218843283991;

      cam.distCoeffs = cv::Mat:: zeros(1,1,CV_32F); // We dont want to undistort im ORBSLAM so we pass dummy matrix

      cv::Mat m_undistortion_map1;
      cv::Mat m_undistortion_map2;

      cv::cuda::GpuMat m_undistortion_map_1;
      cv::cuda::GpuMat m_undistortion_map_2;
      cv::cuda::GpuMat m_undistorted_image_gpu = cv::cuda::GpuMat(1536, 2048, CV_8UC3);

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


      cam.fps        = 10;
      cam.orig_width      = static_cast<int>(2048*resize_factor);
      cam.orig_height     = static_cast<int>(1536*resize_factor);

      //1.0
      cam.new_width      = static_cast<int>(2048*resize_factor);
      cam.new_height     = static_cast<int>(1536*resize_factor);
      cam.isRGB      = false; // BGR

      ORB_SLAM3::OrbParameters orb{};
      orb.nFeatures   = 6000;
      orb.nFastFeatures = 16000;
      orb.nLevels     = 1;
      orb.scaleFactor = 1.8;
      orb.minThFast   = 80;
      orb.iniThFast   = 100;

      //F6
      ORB_SLAM3::ImuParameters imu;
      imu.accelWalk  = 0.007579860836224204; //x2000
      imu.gyroWalk   = 0.00352677789580351; //x2000
      imu.noiseAccel =  0.003138444640779682; //x20
      imu.noiseGyro  = 0.003278894143880944; // x20
      imu.InsertKFsWhenLost = false;

      cv::Mat cv_Tbc = cv::Mat::zeros(4,4,CV_32F);

      cv_Tbc.at<float>(0,0) =   0.0147249;
      cv_Tbc.at<float>(0,1) =   0.00100526;
      cv_Tbc.at<float>(0,2) =   0.99989108;
      cv_Tbc.at<float>(0,3) =   0.0504444;

      cv_Tbc.at<float>(1,0) =   -0.00472214;
      cv_Tbc.at<float>(1,1) =   -0.99998827;
      cv_Tbc.at<float>(1,2) =   0.0010749;
      cv_Tbc.at<float>(1,3) =   -0.01192971;

      cv_Tbc.at<float>(2,0) =   0.99988043;
      cv_Tbc.at<float>(2,1) =   -0.00473745;
      cv_Tbc.at<float>(2,2) =   -0.01471998;
      cv_Tbc.at<float>(2,3) =   -0.01500127;

      cv_Tbc.at<float>(3,0) =   0.0;
      cv_Tbc.at<float>(3,1) =   0.0;
      cv_Tbc.at<float>(3,2) =   0.0;
      cv_Tbc.at<float>(3,3) =   1.0;

      imu.Tbc = cv_Tbc;
      imu.freq = 400.0;

      ORB_SLAM3::LocalMapperParameters local_mapper;
      local_mapper.resetTimeThresh = 10.0;
      local_mapper.minTimeForVIBA1 = 5.0;
      local_mapper.minTimeForVIBA2 = 7.0;
      local_mapper.minTimeForFullBA = 60.0;

      double timeshift_cam_imu = 0.008390335701785497; 

      const int frame_grid_cols = 64;
      const int frame_grid_rows = 48;
      const double clahe_clip_limit = 80.0;
      const int clahe_grid_size = 8;

      // Create SLAM system. It initializes all system threads and gets ready to process frames.
      SLAM_ = std::make_unique<ORB_SLAM3::System>(path_to_vocab_,cam, imu, orb, local_mapper, ORB_SLAM3::System::IMU_MONOCULAR, frame_grid_cols,frame_grid_rows,false, true);
      cout << "SLAM Init" << endl;

      auto sub_image_options = rclcpp::SubscriptionOptions();
      sub_image_options.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      auto sub_imu_options = rclcpp::SubscriptionOptions();
      sub_imu_options.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      imugb_ = std::make_shared<ImuGrabber>(this->get_logger());
      igb_ = std::make_unique<ImageGrabber>(SLAM_.get(),imugb_,bEqual_, timeshift_cam_imu, cam.new_width, cam.new_height, resize_factor,clahe_clip_limit, clahe_grid_size, m_undistortion_map_1, m_undistortion_map_2, m_undistorted_image_gpu, this->get_logger());
      sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>("/bmi088_F6/imu", rclcpp::SensorDataQoS().keep_last(5000), bind(&ImuGrabber::GrabImu, imugb_.get(), placeholders::_1),sub_imu_options);
      sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>("/AIT_Fighter6/down/image", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImageGrabber::GrabImage, igb_.get(), placeholders::_1),sub_image_options);
      sync_thread_ = std::make_unique<std::thread>(&ImageGrabber::SyncWithImu,igb_.get());
    }

    ~SlamNode(){
      cout << "Trigger Shutdown" << endl;
      SLAM_->Shutdown();
      sync_thread_->join();
    }

  private:
    std::string path_to_vocab_;
    bool bEqual_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_;
    std::shared_ptr<ImuGrabber> imugb_;
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

  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 3);
  auto node = std::make_shared<SlamNode>(argv[1],bEqual);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}