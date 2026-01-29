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
      // ORB_SLAM3::CameraParameters cam{};
      // cam.K = cv::Mat::zeros(3,3,CV_32F);
      // cam.K.at<float>(0,0) = 1331.1713955614885;
      // cam.K.at<float>(1,1) = 1329.7824642924002;
      // cam.K.at<float>(0,2) = 1032.7638177042552;
      // cam.K.at<float>(1,2) = 753.7287922955178;
      // cam.K.at<float>(2,2) = 1;

      // cv::Mat distCoeffs  = cv::Mat:: zeros(4,1,CV_32F);
      // distCoeffs.at<float>(0,0) = 0.014127946548806695;
      // distCoeffs.at<float>(1,0) = -0.007081005355739692;
      // distCoeffs.at<float>(2,0) = 0.02477688100909874;
      // distCoeffs.at<float>(3,0) = -0.012940251836647083;

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


      cam.orig_width      = static_cast<int>(2048*resize_factor);
      cam.orig_height     = static_cast<int>(1536*resize_factor);

      //1.0
      cam.new_width      = static_cast<int>(2048*resize_factor);
      cam.new_height     = static_cast<int>(1536*resize_factor);
      cam.isRGB      = false; // BGR

      ORB_SLAM3::OrbParameters orb{};
      orb.nFeatures   = 10000;
      orb.nFastFeatures = 16000;
      orb.nLevels     = 1;
      orb.scaleFactor = 2.0;
      orb.minThFast   = 80;
      orb.iniThFast   = 100;

      // F6
      // ORB_SLAM3::ImuParameters imu;
      // imu.accelWalk  = 0.0003789930418112102; //x1
      // imu.gyroWalk   = 0.000017633889479017548; //x1
      // imu.noiseAccel = 0.015692223203898409; //x10
      // imu.noiseGyro  = 0.001639447071940472; // x10
      // imu.InsertKFsWhenLost = false;

      // F4
      ORB_SLAM3::ImuParameters imu;
      imu.accelWalk  = 0.0006431373218006597; //x1
      imu.gyroWalk   = 0.000018714270991865037; //x1
      imu.noiseAccel = 0.015592957173554883; //x10
      imu.noiseGyro  = 0.0017019559710036963; // x10
      imu.InsertKFsWhenLost = false;

      cv::Mat cv_Tbc = cv::Mat::zeros(4,4,CV_32F);
      
      // F6
      // cv_Tbc.at<float>(0,0) =   0.00475163;
      // cv_Tbc.at<float>(0,1) =   0.01068302;
      // cv_Tbc.at<float>(0,2) =   0.99993165;
      // cv_Tbc.at<float>(0,3) =   0.0685495;

      // cv_Tbc.at<float>(1,0) =   -0.00380494;
      // cv_Tbc.at<float>(1,1) =   -0.9999355;
      // cv_Tbc.at<float>(1,2) =   0.01070114;
      // cv_Tbc.at<float>(1,3) =   -0.00696092;

      // cv_Tbc.at<float>(2,0) =   0.99998147;
      // cv_Tbc.at<float>(2,1) =   -0.00385553;
      // cv_Tbc.at<float>(2,2) =   -0.00471067;
      // cv_Tbc.at<float>(2,3) =   -0.00955084;

      // cv_Tbc.at<float>(3,0) =   0.0;
      // cv_Tbc.at<float>(3,1) =   0.0;
      // cv_Tbc.at<float>(3,2) =   0.0;
      // cv_Tbc.at<float>(3,3) =   1.0;

      // F4
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


      imu.Tbc = cv_Tbc;
      imu.freq = 400.0;

      ORB_SLAM3::LocalMapperParameters local_mapper;
      local_mapper.resetTimeThresh = 500.0;
      local_mapper.minTimeForImuInit = 40.0;
      local_mapper.minTimeForVIBA1 = 50.0;
      local_mapper.minTimeForVIBA2 = 100.0;
      local_mapper.minTimeForFullBA = -1.0;
      local_mapper.thFarPoints = 0.0;
      local_mapper.itsFIBAInit = 5;
      local_mapper.itsFIBA1 = 5;
      
      // F6
      //double timeshift_cam_imu = 0.006882460203406222; 

      // F4 
      double timeshift_cam_imu = 0.00851880502751802;

      ORB_SLAM3::TrackerParameters tracker_settings;
      tracker_settings.frameGridCols = 64;
      tracker_settings.frameGridRows = 48;
      tracker_settings.maxLocalKFCount = 10;
      tracker_settings.featureThresholdForKF = 100;
      tracker_settings.maxFrames = 10;

      const double clahe_clip_limit = 80.0;
      const int clahe_grid_size = 8;

      // Create SLAM system. It initializes all system threads and gets ready to process frames.
      SLAM_ = std::make_shared<ORB_SLAM3::System>(path_to_vocab_,cam, imu, orb, local_mapper, tracker_settings, ORB_SLAM3::System::IMU_MONOCULAR,false, true);
      cout << "SLAM Init" << endl;

      auto sub_image_options = rclcpp::SubscriptionOptions();
      sub_image_options.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      auto sub_imu_options = rclcpp::SubscriptionOptions();
      sub_imu_options.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      imugb_ = std::make_shared<ImuGrabber>(this->get_logger());
      igb_ = std::make_unique<ImageGrabber>(SLAM_,imugb_,bEqual_, timeshift_cam_imu, cam.new_width, cam.new_height, resize_factor,clahe_clip_limit, clahe_grid_size, m_undistortion_map_1, m_undistortion_map_2, m_undistorted_image_gpu,std::chrono::milliseconds(50), this->get_logger());
      sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>("/bmi088_F4/imu", rclcpp::SensorDataQoS().keep_last(500000), bind(&ImuGrabber::GrabImu, imugb_.get(), placeholders::_1),sub_imu_options);
      sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>("/AIT_Fighter4/down/image", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImageGrabber::GrabImage, igb_.get(), placeholders::_1),sub_image_options);
      sync_thread_ = std::make_unique<std::thread>(&ImageGrabber::SyncWithImu,igb_.get());
    }

    ~SlamNode(){
      cout << "Trigger Shutdown" << endl;
      if(!SLAM_->isShutDown())
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
    std::shared_ptr<ORB_SLAM3::System> SLAM_;
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