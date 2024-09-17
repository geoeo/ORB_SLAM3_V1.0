#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>


#include <ORB_SLAM3/System.h>
#include <ORB_SLAM3/ImuTypes.h>
#include <tracy/Tracy.hpp>

using namespace std::chrono_literals;

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
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe, double tshift_cam_imu, float resize_factor, const cv::Mat &undistortion_map1, const cv::Mat& undistortion_map2)
      : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), timeshift_cam_imu(tshift_cam_imu),count(0), img_resize_factor(resize_factor),
        m_undistortion_map1(undistortion_map1), m_undistortion_map2(undistortion_map2){
      }

    void GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    cv::Mat GetImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
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
    cv::Mat m_undistortion_map1;
    cv::Mat m_undistortion_map2;
};

void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  mBufMutex.lock();
  if (!img0Buf.empty())
    img0Buf.pop();
  img0Buf.push(img_msg);
  count = 0;
  

  mBufMutex.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    throw(cv_bridge::Exception("Error converting image!"));
  }
  
  if(cv_ptr->image.type()==0)
  {
    cv::Mat img_undistorted;
    if(!(m_undistortion_map1.empty() || m_undistortion_map2.empty())){
      cv::remap(cv_ptr->image, img_undistorted, m_undistortion_map1, m_undistortion_map2, cv::InterpolationFlags::INTER_CUBIC);
    }
    else{
      img_undistorted = cv_ptr->image;
    }


    if(img_resize_factor != 1.0){
      auto width = cv_ptr->image.cols;
      auto height = cv_ptr->image.rows;
      cv::Size new_im_size = cv::Size(static_cast<int>(width*img_resize_factor),static_cast<int>(height*img_resize_factor));
      cv::Mat im_resize;
      cv::resize(img_undistorted, im_resize, new_im_size);
      return im_resize;
    } else {
      return img_undistorted;
    }
  }
  else
  {
    std::cout << "Error type" << std::endl;
    return cv_ptr->image.clone();
  }
}


void ImageGrabber::SyncWithImu()
{
  double init_ts = 0;
  while(1)
  {
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
      im = GetImage(img0Buf.front());
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
      if(mbClahe)
        mClahe->apply(im,im);

      if(!vImuMeas.empty() && init_ts != 0){
        std::cout << "IMU meas size: " << vImuMeas.size() << std::endl;
        auto pose_flag_pair = mpSLAM->TrackMonocular(im,tIm,vImuMeas);
        Sophus::Matrix4f pose = pose_flag_pair.first.matrix();
        bool ba_complete_for_frame = pose_flag_pair.second;
        vImuMeas.clear();
        auto timestamps =  mpSLAM->GetScaleChangeTimestamps();
        cout << "BA completed: " << mpSLAM->InertialBACompleted() << endl;
        cout << "BA completed for frame: " << ba_complete_for_frame << endl;
        cout << "Scale Factor: " << mpSLAM->GetScaleFactor() << endl;
        cout << "Current ts: " << tIm << endl;
        for(auto ts : timestamps)
          cout << " ts: " << ts;
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

      // float resize_factor = 0.8;

      // Eve
      // ORB_SLAM3::CameraParameters cam{};
      // cam.K = cv::Mat::zeros(3,3,CV_32F);
      // cam.K.at<float>(0,0) = 1388.9566234253055;
      // cam.K.at<float>(1,1) = 1389.860526555566;
      // cam.K.at<float>(0,2) = 944.8106061888452;
      // cam.K.at<float>(1,2) = 602.163082548295;
      // cam.K.at<float>(2,2) = 1;

      // cv::Mat distCoeffs = cv::Mat:: zeros(4,1,CV_32F);
      // distCoeffs.at<float>(0) = -0.19819316734046494;
      // distCoeffs.at<float>(1) = 0.08670622892662087;
      // distCoeffs.at<float>(2) = -0.0008400222221221046;
      // distCoeffs.at<float>(3) = 0.0005366633601752759;

      // cam.distCoeffs = cv::Mat:: zeros(1,1,CV_32F); // We dont want to undistort im ORBSLAM so we pass dummy matrix
      // //cam.distCoeffs = distCoeffs.clone();

      // cv::Mat m_undistortion_map1;
      // cv::Mat m_undistortion_map2;

      // //TODO: fisheye
      // cv::initUndistortRectifyMap(cam.K,
      //                   distCoeffs,
      //                   cv::Mat_<double>::eye(3, 3),
      //                   cam.K,
      //                   cv::Size(1920, 1200),
      //                   CV_16SC2,
      //                   m_undistortion_map1,
      //                   m_undistortion_map2);

      // cam.K.at<float>(0,0) *= resize_factor;
      // cam.K.at<float>(1,1) *= resize_factor;
      // cam.K.at<float>(0,2) *= resize_factor;
      // cam.K.at<float>(1,2) *= resize_factor;

      // cam.fps        = 8;
      // cam.orig_width      = static_cast<int>(1920*resize_factor);
      // cam.orig_height     = static_cast<int>(1200*resize_factor);

      // //1.0
      // cam.new_width      = static_cast<int>(1920*resize_factor);
      // cam.new_height     = static_cast<int>(1200*resize_factor);
      // cam.isRGB      = false; // BGR

      // ORB_SLAM3::OrbParameters orb{};
      // orb.nFeatures   = 2500;
      // orb.nLevels     = 8;
      // orb.scaleFactor = 1.2;
      // orb.minThFast   = 5;
      // orb.iniThFast   = 15;
      // orb.gridCount = 96;

      // ORB_SLAM3::ImuParameters m_imu;
      // m_imu.accelWalk  = 0.000288252284411655; // x10
      // m_imu.gyroWalk   = 0.0000162566517589794; // x10
      // m_imu.noiseAccel =  0.007302644894222149; //x5
      // m_imu.noiseGyro  = 0.0009336557780556743; // x5

      // m_imu.InsertKFsWhenLost = false;

      // cv::Mat cv_Tbc = cv::Mat::zeros(4,4,CV_32F);

      // cv_Tbc.at<float>(0,0) =   -0.00345318;
      // cv_Tbc.at<float>(0,1) =   -0.05123323;
      // cv_Tbc.at<float>(0,2) =   -0.99868075;
      // cv_Tbc.at<float>(0,3) =   -0.0605664;

      // cv_Tbc.at<float>(1,0) =   -0.00013874;
      // cv_Tbc.at<float>(1,1) =   -0.99868667;
      // cv_Tbc.at<float>(1,2) =   0.05123401;
      // cv_Tbc.at<float>(1,3) =   -0.01364959;

      // cv_Tbc.at<float>(2,0) =   -0.99999403;
      // cv_Tbc.at<float>(2,1) =   0.00031548;
      // cv_Tbc.at<float>(2,2) =   0.00344154;
      // cv_Tbc.at<float>(2,3) =   -0.01763391;

      // cv_Tbc.at<float>(3,0) =   0.0;
      // cv_Tbc.at<float>(3,1) =   0.0;
      // cv_Tbc.at<float>(3,2) =   0.0;
      // cv_Tbc.at<float>(3,3) =   1.0;

      // m_imu.Tbc = cv_Tbc;
      // m_imu.freq = 200.0;

      // double timeshift_cam_imu = -0.013490768586712722; // EvE



      float resize_factor = 0.8;

      // F1
      ORB_SLAM3::CameraParameters cam{};
      cam.K = cv::Mat::zeros(3,3,CV_32F);
      cam.K.at<float>(0,0) = 1341.3908261080117;
      cam.K.at<float>(1,1) = 1339.801008398156;
      cam.K.at<float>(0,2) = 1025.4354831702265;
      cam.K.at<float>(1,2) = 748.2339952852544;
      cam.K.at<float>(2,2) = 1;

      cv::Mat distCoeffs  = cv::Mat:: zeros(4,1,CV_32F);
      distCoeffs.at<float>(0,0) = -0.020898721110400503;
      distCoeffs.at<float>(1,0) = 0.10004887885496858;
      distCoeffs.at<float>(2,0) = -0.12205916600511264;
      distCoeffs.at<float>(3,0) = 0.05976140792758462;

      cam.distCoeffs = cv::Mat:: zeros(1,1,CV_32F); // We dont want to undistort im ORBSLAM so we pass dummy matrix

      cv::Mat m_undistortion_map1;
      cv::Mat m_undistortion_map2;

      //TODO: fisheye
      cv::fisheye::initUndistortRectifyMap(cam.K,
                        distCoeffs,
                        cv::Mat_<double>::eye(3, 3),
                        cam.K,
                        cv::Size(2048, 1536),
                        CV_16SC2,
                        m_undistortion_map1,
                        m_undistortion_map2);

      cam.K.at<float>(0,0) *= resize_factor;
      cam.K.at<float>(1,1) *= resize_factor;
      cam.K.at<float>(0,2) *= resize_factor;
      cam.K.at<float>(1,2) *= resize_factor;


      cam.fps        = 8;
      cam.orig_width      = static_cast<int>(2048*resize_factor);
      cam.orig_height     = static_cast<int>(1536*resize_factor);

      //1.0
      cam.new_width      = static_cast<int>(2048*resize_factor);
      cam.new_height     = static_cast<int>(1536*resize_factor);
      cam.isRGB      = false; // BGR

      ORB_SLAM3::OrbParameters orb{};
      orb.nFeatures   = 2500;
      orb.nLevels     = 8;
      orb.scaleFactor = 1.2;
      orb.minThFast   = 5;
      orb.iniThFast   = 15;
      orb.gridCount = 96;

      ORB_SLAM3::ImuParameters m_imu;
      m_imu.accelWalk  = 0.00047746677530925284; // x10
      m_imu.gyroWalk   = 0.00001637752; //x10
      m_imu.noiseAccel =  0.01544130852; // x10
      m_imu.noiseGyro  = 0.00171257248; // x10

      m_imu.InsertKFsWhenLost = false;

      cv::Mat cv_Tbc = cv::Mat::zeros(4,4,CV_32F);

      cv_Tbc.at<float>(0,0) =   0.00555971;
      cv_Tbc.at<float>(0,1) =   0.02936748;
      cv_Tbc.at<float>(0,2) =   0.99955322;
      cv_Tbc.at<float>(0,3) =   0.05443226;

      cv_Tbc.at<float>(1,0) =   0.00321125;
      cv_Tbc.at<float>(1,1) =   -0.99956404;
      cv_Tbc.at<float>(1,2) =   0.02934994;
      cv_Tbc.at<float>(1,3) =   -0.01588301;

      cv_Tbc.at<float>(2,0) =   0.99997939;
      cv_Tbc.at<float>(2,1) =   0.00304664;
      cv_Tbc.at<float>(2,2) =   -0.00565159;
      cv_Tbc.at<float>(2,3) =   -0.02002308;

      cv_Tbc.at<float>(3,0) =   0.0;
      cv_Tbc.at<float>(3,1) =   0.0;
      cv_Tbc.at<float>(3,2) =   0.0;
      cv_Tbc.at<float>(3,3) =   1.0;

      m_imu.Tbc = cv_Tbc;
      m_imu.freq = 400.0;

      double timeshift_cam_imu = 0.008684532573338512; // F1

      // Create SLAM system. It initializes all system threads and gets ready to process frames.
      SLAM_ = std::make_unique<ORB_SLAM3::System>(path_to_vocab_,cam,m_imu, orb, ORB_SLAM3::System::IMU_MONOCULAR, false, true);
      cout << "SLAM Init" << endl;

      igb_ = std::make_unique<ImageGrabber>(SLAM_.get(),&imugb_,bEqual_, timeshift_cam_imu, resize_factor, m_undistortion_map1, m_undistortion_map2);
      sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>("/bmi088/imu", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImuGrabber::GrabImu, &imugb_, placeholders::_1));
      //sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>("/down/genicam_0/image", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImageGrabber::GrabImage, igb_.get(), placeholders::_1));
      sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>("/AIT_Fighter5/down/image", rclcpp::SensorDataQoS().keep_last(1000), bind(&ImageGrabber::GrabImage, igb_.get(), placeholders::_1));
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