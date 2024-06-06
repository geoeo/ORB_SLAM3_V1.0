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

using namespace std::chrono_literals;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::msg::Imu &imu_msg);

    queue<sensor_msgs::msg::Imu> imuBuf;
    std::mutex mBufMutex;
};

void ImuGrabber::GrabImu(const sensor_msgs::msg::Imu &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bClahe, double tshift_cam_imu, uint64_t fps_fac, float resize_factor)
      : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), timeshift_cam_imu(tshift_cam_imu),fps_factor(fps_fac),count(0), img_resize_factor(resize_factor) {}

    void GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    cv::Mat GetImage(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::msg::Image::ConstSharedPtr> img0Buf;
    std::mutex mBufMutex;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(4, 4));
    double timeshift_cam_imu;
    uint64_t fps_factor;
    uint64_t count;
    float img_resize_factor;
};

void ImageGrabber::GrabImage(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
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

cv::Mat ImageGrabber::GetImage(const sensor_msgs::msg::Image::ConstSharedPtr  &img_msg)
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
    auto width = cv_ptr->image.cols;
    auto height = cv_ptr->image.rows;
    cv::Size new_im_size = cv::Size(static_cast<int>(width*img_resize_factor),static_cast<int>(height*img_resize_factor));
    cv::Mat im_resize;
    cv::resize(cv_ptr->image, im_resize, new_im_size);
    return im_resize;
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
      auto ros_image_ts_front = rclcpp::Time(img0Buf.front()->header.stamp);
      mpImuGb->mBufMutex.lock();
      auto ros_imu_ts_back = rclcpp::Time(mpImuGb->imuBuf.back().header.stamp);
      mpImuGb->mBufMutex.unlock();
      tIm = ros_image_ts_front.seconds() + timeshift_cam_imu;
      if(tIm>ros_imu_ts_back.seconds())
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
        auto ros_imu_ts_front = rclcpp::Time(mpImuGb->imuBuf.front().header.stamp);
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!mpImuGb->imuBuf.empty() && ros_imu_ts_front.seconds()<=tIm)
        {
          double t = ros_imu_ts_front.seconds();
          cv::Point3f acc(mpImuGb->imuBuf.front().linear_acceleration.x, mpImuGb->imuBuf.front().linear_acceleration.y, mpImuGb->imuBuf.front().linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front().angular_velocity.x, mpImuGb->imuBuf.front().angular_velocity.y, mpImuGb->imuBuf.front().angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      if(mbClahe)
        mClahe->apply(im,im);

      //std::cout << "IMU meas size: " << vImuMeas.size() << std::endl;
      if(!vImuMeas.empty()){
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

class MinimalPublisher : public rclcpp::Node
{
  public:
    MinimalPublisher()
    : Node("minimal_publisher"), count_(0)
    {
      publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
      timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

  private:
    void timer_callback()
    {
      auto message = std_msgs::msg::String();
      message.data = "Hello, world! " + std::to_string(count_++);
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
      publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}