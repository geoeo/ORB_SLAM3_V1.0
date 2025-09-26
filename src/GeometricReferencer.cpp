#include <GeometricReferencer.hpp>

using namespace std;
using namespace ORB_SLAM3;

GeometricReferencer::GeometricReferencer(int min_nrof_frames)
: m_is_initialized(false),
  m_min_nrof_frames(min_nrof_frames)
{
}

void GeometricReferencer::clear()
{
  m_is_initialized = false;
  m_spatials.clear();
}

bool GeometricReferencer::isInitialized()
{
  return m_is_initialized;
}

optional<pair<Sophus::SE3d, double>> GeometricReferencer::init(const vector<ORB_SLAM3::KeyFrame*> &frames)
{

  if (m_is_initialized)
    return nullopt;
  
  if (frames.size() < m_min_nrof_frames)
    return nullopt;
  
  auto [pose, scale] = update(frames); 

  m_is_initialized = true;
  return {{pose, scale}};
}

pair<Sophus::SE3d, double> GeometricReferencer::update(const vector<ORB_SLAM3::KeyFrame*> &frames)
{ 

  for (const auto &f : frames){
    const auto gnss_position = f->GetGNSSPosition();
    const auto gnss_pose = Sophus::SE3d(Eigen::Matrix3d::Identity(), gnss_position.cast<double>());
    m_spatials.pop_front();
    m_spatials.push_back({gnss_pose, f->GetPoseInverse().cast<double>()});
  }

  return estimateGeorefTransform(m_spatials, Sophus::SE3d(), true);
}

pair<Sophus::SE3d, double> GeometricReferencer::estimateGeorefTransform(const std::deque<pair<Sophus::SE3d, Sophus::SE3d>> &spatials, const Sophus::SE3d &T_w2g_init, bool estimate_scale)
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

    auto e_gis_0 = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    auto e_gis_x = T_rec2g_gis* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    auto e_gis_y = T_rec2g_gis* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    auto e_gis_z = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    dst_points.col(j) << e_gis_x(0), e_gis_x(1), e_gis_x(2);
    dst_points.col(j+1) << e_gis_y(0), e_gis_y(1), e_gis_y(2);
    dst_points.col(j+2) << e_gis_z(0), e_gis_z(1), e_gis_z(2);
    dst_points.col(j+3) << e_gis_0(0), e_gis_0(1), e_gis_0(2);

    auto T_c2w = spatials[i].second;
    auto T_c2g = T_w2g_init*T_c2w;

    auto e_vis_0 = T_c2g* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    auto e_vis_x = T_c2g* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    auto e_vis_y = T_c2g* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    auto e_vis_z = T_c2g* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    src_points.col(j) << e_vis_x(0), e_vis_x(1), e_vis_x(2);
    src_points.col(j+1) << e_vis_y(0), e_vis_y(1), e_vis_y(2);
    src_points.col(j+2) << e_vis_z(0), e_vis_z(1), e_vis_z(2);
    src_points.col(j+3 ) << e_vis_0(0), e_vis_0(1), e_vis_0(2);
  }
  
  //Estimates the aligning transformation from camera to gnss coordinate system
  Eigen::Matrix4d T_g2receiver_refine = Eigen::umeyama(src_points, dst_points, estimate_scale);

  auto rotation_matrix = T_g2receiver_refine.block<3,3>(0,0);

  auto sx = rotation_matrix.col(0).norm();
  auto sy = rotation_matrix.col(1).norm();
  auto sz = rotation_matrix.col(2).norm();

  auto scale = (sx + sy + sz)/3;

  return {Sophus::SE3d(T_g2receiver_refine), scale};
}
