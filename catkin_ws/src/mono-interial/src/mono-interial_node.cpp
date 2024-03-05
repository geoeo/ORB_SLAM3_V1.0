#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include <ORB_SLAM3/System.h>
#include <ORB_SLAM3/ImuTypes.h>

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe, double tshift_cam_imu, uint64_t fps_fac)
      : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), timeshift_cam_imu(tshift_cam_imu),fps_factor(fps_fac),count(0) {}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> img0Buf;
    std::mutex mBufMutex;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(4, 4));
    double timeshift_cam_imu;
    uint64_t fps_factor;
    uint64_t count;
};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "Mono_Inertial");
  ros::NodeHandle n("~");
  n.setParam("/use_sim_time", true);
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  bool bEqual = false;
  if(argc < 3 || argc > 4)
  {
    cerr << endl << "Usage: rosrun mono_inertial node path_to_vocabulary path_to_settings [do_equalize]" << endl;
    ros::shutdown();
    return 1;
  }


  if(argc==4)
  {
    std::string sbEqual(argv[3]);
    if(sbEqual == "true")
      bEqual = true;
  }

  // Eve
  ORB_SLAM3::CameraParameters cam{};
  cam.K = cv::Mat::zeros(3,3,CV_32F);
  cam.K.at<float>(0,0) = 1388.9566234253055;
  cam.K.at<float>(1,1) = 1389.860526555566;
  cam.K.at<float>(0,2) = 944.8106061888452;
  cam.K.at<float>(1,2) = 602.163082548295;
  cam.K.at<float>(2,2) = 1;


  cam.distCoeffs = cv::Mat::zeros(4,1,CV_32F);
  cam.distCoeffs.at<float>(0,0) = -0.19819316734046494;
  cam.distCoeffs.at<float>(1,0) = 0.08670622892662087;
  cam.distCoeffs.at<float>(2,0) = -0.0008400222221221046;
  cam.distCoeffs.at<float>(3,0) = 0.0005366633601752759;

  cam.fps        = 17;
  cam.width      = 768;
  cam.height     = 480;
  cam.isRGB      = false; // BGR

  ORB_SLAM3::OrbParameters orb{};
  orb.nFeatures   = 2000;
  orb.nLevels     = 7;
  orb.scaleFactor = 1.2;
  orb.minThFast   = 5;
  orb.iniThFast   = 15;

  ORB_SLAM3::ImuParameters m_imu;
  m_imu.accelWalk  = 3.0000e-3;
  m_imu.gyroWalk   = 1.9393e-05;
  m_imu.noiseAccel = 2.0000e-3;
  m_imu.noiseGyro  = 1.6968e-04;

  cv::Mat cv_Tbc = cv::Mat::zeros(4,4,CV_32F);

  cv_Tbc.at<float>(0,0) =   -0.00345318;
  cv_Tbc.at<float>(0,1) =   -0.05123323;
  cv_Tbc.at<float>(0,2) =   -0.99868075;
  cv_Tbc.at<float>(0,3) =   -0.0605664;

  cv_Tbc.at<float>(1,0) =   -0.00013874;
  cv_Tbc.at<float>(1,1) =   -0.99868667;
  cv_Tbc.at<float>(1,2) =   -0.05123401;
  cv_Tbc.at<float>(1,3) =   -0.01364959;

  cv_Tbc.at<float>(2,0) =   -0.99999403;
  cv_Tbc.at<float>(2,1) =   -0.00031548;
  cv_Tbc.at<float>(2,2) =   -0.00344154;
  cv_Tbc.at<float>(2,3) =   -0.01763391;

  cv_Tbc.at<float>(3,0) =   0.0;
  cv_Tbc.at<float>(3,1) =   0.0;
  cv_Tbc.at<float>(3,2) =   0.0;
  cv_Tbc.at<float>(3,3) =   1.0;

  m_imu.Tbc = cv_Tbc;
  m_imu.freq       = 200.0;




  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MONOCULAR,true);
  //ORB_SLAM3::System SLAM(argv[1],cam,m_imu, orb, ORB_SLAM3::System::MONOCULAR, true, true);


  //double timeshift_cam_imu = 0.0021434982252719545; //Kaist
  //uint64_t fps_factor = 3; //Kaist (10fps)

  //double timeshift_cam_imu = 0.0; //Euroc
  double timeshift_cam_imu = -0.013490768586712722; // EvE


  uint64_t fps_factor = 1;

  ImuGrabber imugb;
  ImageGrabber igb(&SLAM,&imugb,bEqual, timeshift_cam_imu, fps_factor); // TODO

    // Maximum delay, 5 seconds
  ros::Subscriber sub_imu = n.subscribe("/bmi088/imu", 1000, &ImuGrabber::GrabImu, &imugb); 
  ros::Subscriber sub_img0 = n.subscribe("/down/genicam_0/image", 1000, &ImageGrabber::GrabImage,&igb);
  //Euroc
 // ros::Subscriber sub_imu = n.subscribe("/imu0", 100, &ImuGrabber::GrabImu, &imugb); 
 //ros::Subscriber sub_img0 = n.subscribe("/cam0/image_raw", 1000, &ImageGrabber::GrabImage,&igb);

  

  std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);

  ros::spin();

   return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &img_msg)
{

  mBufMutex.lock();
  count++;
  if (!img0Buf.empty())
    img0Buf.pop();
  if(count == fps_factor){ //hardcoded fps factor Kaist
    img0Buf.push(img_msg);
    count = 0;
  } 

  mBufMutex.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  
  if(cv_ptr->image.type()==0)
  {
    return cv_ptr->image.clone();
  }
  else
  {
    std::cout << "Error type" << std::endl;
    return cv_ptr->image.clone();
  }
}

void ImageGrabber::SyncWithImu()
{
  while(1)
  {
    cv::Mat im;
    double tIm = 0;
    if (!img0Buf.empty()&&!mpImuGb->imuBuf.empty())
    {
      tIm = img0Buf.front()->header.stamp.toSec() + timeshift_cam_imu;
      if(tIm>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;
      {
      this->mBufMutex.lock();
      im = GetImage(img0Buf.front());
      img0Buf.pop();
      this->mBufMutex.unlock();
      }

      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tIm)
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      if(mbClahe)
        mClahe->apply(im,im);

      std::cout << "IMU meas size: " << vImuMeas.size() << std::endl;
      if(!vImuMeas.empty()){
        Sophus::Matrix4f pose = mpSLAM->TrackMonocular(im,tIm,vImuMeas).matrix();
        cout << "ORB POSE:" << endl;
        cout << pose(0,0) << ", " << pose(0,1) << ", " << pose(0,2) << ", " << pose(0,3) << endl;
        cout << pose(1,0) << ", " << pose(1,1) << ", " << pose(1,2) << ", " << pose(1,3) << endl;
        cout << pose(2,0) << ", " << pose(2,1) << ", " << pose(2,2) << ", " << pose(2,3) << endl;
        cout << pose(3,0) << ", " << pose(3,1) << ", " << pose(3,2) << ", " << pose(3,3) << endl;
      }

    }

    //std::chrono::milliseconds tSleep(1);
    //std::this_thread::sleep_for(tSleep);
  }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}

