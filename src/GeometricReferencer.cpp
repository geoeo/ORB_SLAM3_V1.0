#include <GeometricReferencer.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace ORB_SLAM3;

GeometricReferencer::GeometricReferencer(double th_error, int min_nrof_frames)
: m_is_initialized(false),
  m_prev_nrof_unique(0),
  m_T_w2g(Sophus::SE3d()),
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
  m_T_w2g = Sophus::SE3d();
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

Sophus::SE3d GeometricReferencer::getTransform() {
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

  for (const auto &f : frames){
    const auto gnss_position = f->GetGNSSPosition();
    const auto gnss_pose = Sophus::SE3d(Eigen::Matrix3d::Identity(), gnss_position.cast<double>());
    m_spatials.push_back({gnss_pose, f->GetPoseInverse().cast<double>()});
  }


  auto T_w2g = estimateGeorefTransform(m_spatials, Sophus::SE3d(), true);
  auto rotation_matrix = T_w2g.rotationMatrix();

  auto sx = rotation_matrix.col(0).norm();
  auto sy = rotation_matrix.col(1).norm();
  auto sz = rotation_matrix.col(2).norm();

  m_mutex_transform.lock();
  m_scale = (sx + sy + sz)/3;

  rotation_matrix.col(0) /= m_scale;
  rotation_matrix.col(1) /= m_scale;
  rotation_matrix.col(2) /= m_scale;

//   cv::Mat R = T_w2g.rowRange(0,3).colRange(0,3);
// //   cv::Mat R_corr = Frame::OptimalCorrectionOfRotation(R);
// //   T_w2g.rowRange(0,3).colRange(0,3) = R_corr;



  m_T_w2g = Sophus::SE3d(rotation_matrix,T_w2g.translation());
  m_mutex_transform.unlock();
}

Sophus::SE3d GeometricReferencer::estimateGeorefTransform(const std::deque<pair<Sophus::SE3d, Sophus::SE3d>> &spatials, const Sophus::SE3d &T_w2g_init, bool estimate_scale)
{
  // First define basic eigen variables
  const auto measurements = 4;
  auto nrof_points = spatials.size()*measurements;
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> src_points(3, nrof_points);
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> dst_points(3, nrof_points);

  int i, j;
  for (i = 0, j = 0; i < spatials.size(); ++i, j+=measurements)
  {
    // The measurements of the gnss receiver
    auto T_rec2g_gis = spatials[i].first;

    Eigen::Vector4d e_gis_0 = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d e_gis_x = T_rec2g_gis* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d e_gis_y = T_rec2g_gis* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    Eigen::Vector4d e_gis_z = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    dst_points.col(j) << e_gis_x(0), e_gis_x(1), e_gis_x(2);
    dst_points.col(j+1) << e_gis_y(0), e_gis_y(1), e_gis_y(2);
    dst_points.col(j+2) << e_gis_z(0), e_gis_z(1), e_gis_z(2);
    dst_points.col(j+3) << e_gis_0(0), e_gis_0(1), e_gis_0(2);

    auto T_c2w = spatials[i].second;
    auto T_c2g = T_w2g_init*T_c2w;

    Eigen::Vector4d e_vis_0 = T_c2g* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d e_vis_x = T_c2g* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    Eigen::Vector4d e_vis_y = T_c2g* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    Eigen::Vector4d e_vis_z = T_c2g* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    src_points.col(j) << e_vis_x(0), e_vis_x(1), e_vis_x(2);
    src_points.col(j+1) << e_vis_y(0), e_vis_y(1), e_vis_y(2);
    src_points.col(j+2) << e_vis_z(0), e_vis_z(1), e_vis_z(2);
    src_points.col(j+3 ) << e_vis_0(0), e_vis_0(1), e_vis_0(2);
  }
  
  //Estimates the aligning transformation from camera to gnss coordinate system
  Eigen::Matrix4d T_g2receiver_refine = Eigen::umeyama(src_points, dst_points, estimate_scale);
  return Sophus::SE3d(T_g2receiver_refine);
}
