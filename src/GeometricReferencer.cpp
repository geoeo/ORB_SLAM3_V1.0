#include <GeometricReferencer.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace ORB_SLAM3;

GeometricReferencer::GeometricReferencer(double th_error, int min_nrof_frames)
: m_is_initialized(false),
  m_prev_nrof_unique(0),
  m_T_w2g(cv::Mat::eye(4,4,CV_64F)),
  m_scale(0.0),
  m_th_error(th_error),
  m_error(0.0),
  m_min_nrof_frames(min_nrof_frames)
{
}

void GeometricReferencer::clear()
{
  m_is_initialized = false;
  m_spatials.clear();
  m_T_w2g = cv::Mat::eye(4,4,CV_64F);
  m_scale = 0.0;
  m_prev_nrof_unique = 0;
}

bool GeometricReferencer::isInitialized()
{
  return m_is_initialized;
}

double GeometricReferencer::getScale() {
  unique_lock<mutex> lock(m_mutex_transform);
  return m_scale;
}

cv::Mat GeometricReferencer::getTransform() {
  unique_lock<mutex> lock(m_mutex_transform);
  return m_T_w2g;
}

bool GeometricReferencer::init(const deque<ORB_SLAM3::KeyFrame*> &frames)
{

  if (m_is_initialized)
    return false;
  

  if (frames.size() < m_min_nrof_frames)
    return false;
  
  update(frames); 
  m_is_initialized = true;
  return m_is_initialized;
}

void GeometricReferencer::update(const std::deque<ORB_SLAM3::KeyFrame*> &frames)
{ 

  for (const auto &f : frames)
    m_spatials.push_back({f->getGNSSPose(), f->getVisualPose()});
  
  cv::Mat T_w2g = estimateGeorefTransform(m_spatials, cv::Mat::eye(4, 4, CV_64F), true);

  auto c1 = T_w2g.rowRange(0,3).colRange(0,1);
  auto c2 = T_w2g.rowRange(0,3).colRange(1,2);
  auto c3 = T_w2g.rowRange(0,3).colRange(2,3);

  auto sx = cv::norm(c1);
  auto sy = cv::norm(c2);
  auto sz = cv::norm(c3);

  m_mutex_transform.lock();
  m_scale = (sx + sy + sz)/3;

  T_w2g.rowRange(0,3).colRange(0,1) /= m_scale;
  T_w2g.rowRange(0,3).colRange(1,2) /= m_scale;
  T_w2g.rowRange(0,3).colRange(2,3) /= m_scale;

  cv::Mat R = T_w2g.rowRange(0,3).colRange(0,3);
//   cv::Mat R_corr = Frame::OptimalCorrectionOfRotation(R);
//   T_w2g.rowRange(0,3).colRange(0,3) = R_corr;



  m_T_w2g = T_w2g.clone();
  m_mutex_transform.unlock();
}

cv::Mat GeometricReferencer::estimateGeorefTransform(const std::deque<pair<cv::Mat, cv::Mat>> &spatials, const cv::Mat &T_w2g_init, bool estimate_scale)
{
  // First define basic eigen variables
  const auto measurements = 4;
  auto nrof_points = spatials.size()*measurements;
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> src_points(3, nrof_points);
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> dst_points(3, nrof_points);

  int i, j;
  cv::Mat hom_row = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 1.0);
  for (i = 0, j = 0; i < spatials.size(); ++i, j+=measurements)
  {
    // The measurements of the gnss receiver
    cv::Mat T_rec2g_gis = spatials[i].first.clone();
    T_rec2g_gis.push_back(hom_row);

    cv::Mat e_gis_0, e_gis_x, e_gis_y, e_gis_z;
    e_gis_0 = T_rec2g_gis* (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 1.0);
    e_gis_x = T_rec2g_gis* (cv::Mat_<double>(4, 1) << 1.0, 0.0, 0.0, 1.0);
    e_gis_y = T_rec2g_gis* (cv::Mat_<double>(4, 1) << 0.0, 1.0, 0.0, 1.0);
    e_gis_z = T_rec2g_gis* (cv::Mat_<double>(4, 1) << 0.0, 0.0, 1.0, 1.0);

    dst_points.col(j) << e_gis_x.at<double>(0), e_gis_x.at<double>(1), e_gis_x.at<double>(2);
    dst_points.col(j+1) << e_gis_y.at<double>(0), e_gis_y.at<double>(1), e_gis_y.at<double>(2);
    dst_points.col(j+2) << e_gis_z.at<double>(0), e_gis_z.at<double>(1), e_gis_z.at<double>(2);
    dst_points.col(j+3) << e_gis_0.at<double>(0), e_gis_0.at<double>(1), e_gis_0.at<double>(2);

    cv::Mat T_c2w = spatials[i].second.clone();
    T_c2w.push_back(hom_row);
    cv::Mat T_c2g = T_w2g_init*T_c2w;
    T_c2g.push_back(hom_row);


    cv::Mat e_vis_0, e_vis_x, e_vis_y, e_vis_z;

    e_vis_0 = T_c2g* (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 1.0);
    e_vis_x = T_c2g* (cv::Mat_<double>(4, 1) << 1.0, 0.0, 0.0, 1.0);
    e_vis_y = T_c2g* (cv::Mat_<double>(4, 1) << 0.0, 1.0, 0.0, 1.0);
    e_vis_z = T_c2g* (cv::Mat_<double>(4, 1) << 0.0, 0.0, 1.0, 1.0);


    src_points.col(j) << e_vis_x.at<double>(0), e_vis_x.at<double>(1), e_vis_x.at<double>(2);
    src_points.col(j+1) << e_vis_y.at<double>(0), e_vis_y.at<double>(1), e_vis_y.at<double>(2);
    src_points.col(j+2) << e_vis_z.at<double>(0), e_vis_z.at<double>(1), e_vis_z.at<double>(2);
    src_points.col(j+3 ) << e_vis_0.at<double>(0), e_vis_0.at<double>(1), e_vis_0.at<double>(2);
  }
  
  //Estimates the aligning transformation from camera to gnss coordinate system
  Eigen::Matrix4d T_g2receiver_refine = Eigen::umeyama(src_points, dst_points, estimate_scale);
  cv::Mat T_g2receiver_refine_cv(4, 4, CV_64F);
  cv::eigen2cv(T_g2receiver_refine,T_g2receiver_refine_cv);
  return T_g2receiver_refine_cv;
}
