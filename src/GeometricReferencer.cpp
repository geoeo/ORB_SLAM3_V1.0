#include <GeometricReferencer.hpp>
#include <Verbose.h>
#include <algorithm>

using namespace std;
using namespace ORB_SLAM3;

GeometricReferencer::GeometricReferencer(int min_nrof_frames)
: m_is_initialized(false),
  m_min_nrof_frames(min_nrof_frames),
  mTgw_current(Sophus::Sim3d()),
  m_georefed_kfs_count(0)
{
}

void GeometricReferencer::clear()
{
  m_is_initialized = false;
  mTgw_current = Sophus::Sim3d();
  unique_lock<mutex> lock(mMutexFrames);
  m_latest_frames_to_georef = {};
  m_georefed_kfs_count = 0;
}

void GeometricReferencer::clearFrames()
{
  unique_lock<mutex> lock(mMutexFrames);
  m_latest_frames_to_georef = {};
  m_georefed_kfs_count = 0;
}

void GeometricReferencer::updateGeorefKFsCount(size_t count)
{
  unique_lock<mutex> lock(mMutexFrames);
  m_georefed_kfs_count += count;
}

bool GeometricReferencer::isInitialized() const
{
  return m_is_initialized;
}

Sophus::Sim3d GeometricReferencer::getCurrentTransform()
{
  unique_lock<mutex> lock(mMutexTransform);
  return mTgw_current; 
}

void GeometricReferencer::addKeyFrame(KeyFrame* kf){
  unique_lock<mutex> lock(mMutexFrames);
    if(m_latest_frames_to_georef.size() >= m_min_nrof_frames)
      m_latest_frames_to_georef.pop_front();
    m_latest_frames_to_georef.push_back(kf);
    if(m_georefed_kfs_count > 0)
      --m_georefed_kfs_count;
}

deque<KeyFrame*> GeometricReferencer::getFramesForGeorefEstimation() {
  unique_lock<mutex> lock(mMutexFrames);
  return m_latest_frames_to_georef;
}

vector<KeyFrame*> GeometricReferencer::getFramesWithoutGeoref() {
  unique_lock<mutex> lock(mMutexFrames);
  return m_georefed_kfs_count < m_latest_frames_to_georef.size() ? vector<KeyFrame*>(m_latest_frames_to_georef.cbegin()+m_georefed_kfs_count, m_latest_frames_to_georef.cend()) : vector<KeyFrame*>();
}

optional<Sophus::Sim3d> GeometricReferencer::apply(const std::deque<KeyFrame*> &frames, bool do_update)
{

  if (frames.size() < m_min_nrof_frames)
    return nullopt;

  if (m_is_initialized && !do_update)
    return getCurrentTransform();

  auto pose = update(frames); 

  if(!m_is_initialized)
    m_is_initialized = true;

  return pose;
}

Sophus::Sim3d GeometricReferencer::update(const std::deque<KeyFrame *> &spatials)
{ 
  const auto pose = estimateGeorefTransform(spatials);

  Verbose::PrintMess("FULL Georef function successful", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("Transformation matrix:", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(pose.rotationMatrix()(0,0)) + " " + to_string(pose.rotationMatrix()(0,1)) + " " + to_string(pose.rotationMatrix()(0,2)) + " " + to_string(pose.translation()(0)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(pose.rotationMatrix()(1,0)) + " " + to_string(pose.rotationMatrix()(1,1)) + " " + to_string(pose.rotationMatrix()(1,2)) + " " + to_string(pose.translation()(1)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(pose.rotationMatrix()(2,0)) + " " + to_string(pose.rotationMatrix()(2,1)) + " " + to_string(pose.rotationMatrix()(2,2)) + " " + to_string(pose.translation()(2)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("Scale: " + to_string(pose.scale()), Verbose::VERBOSITY_NORMAL);
  
  // TODO: incremental updates dont seem to work
  unique_lock<mutex> lock(mMutexTransform);
  mTgw_current = pose;
  return mTgw_current;
}

Sophus::Sim3d GeometricReferencer::estimateGeorefTransform(const std::deque<KeyFrame *> &spatials)
{
  // First define basic eigen variables
  const auto measurements = 4;
  auto nrof_points = spatials.size()*measurements;
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> src_points(3, nrof_points);
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> dst_points(3, nrof_points);


  int i, j;
  for (i = 0, j = 0; i < spatials.size(); ++i, j+=measurements)
  {
    const auto f = spatials[i];
    // The measurements of the gnss receiver
    auto T_rec2g_gis = f->GetGNSSCameraPose();

    auto e_gis_0 = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    auto e_gis_x = T_rec2g_gis* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    auto e_gis_y = T_rec2g_gis* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    auto e_gis_z = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    dst_points.col(j) << e_gis_0(0), e_gis_0(1), e_gis_0(2);
    dst_points.col(j+1) << e_gis_x(0), e_gis_x(1), e_gis_x(2);
    dst_points.col(j+2) << e_gis_y(0), e_gis_y(1), e_gis_y(2);
    dst_points.col(j+3) << e_gis_z(0), e_gis_z(1), e_gis_z(2);

    const auto Twc = f->GetPoseInverse();
    const auto Twc_sim3 = Sophus::Sim3d(1.0,Twc.unit_quaternion().cast<double>(), Twc.translation().cast<double>());
    const auto T_c2g = Twc_sim3;

    auto e_vis_0 = T_c2g* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    auto e_vis_x = T_c2g* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    auto e_vis_y = T_c2g* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    auto e_vis_z = T_c2g* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    src_points.col(j ) << e_vis_0(0), e_vis_0(1), e_vis_0(2);
    src_points.col(j+1) << e_vis_x(0), e_vis_x(1), e_vis_x(2);
    src_points.col(j+2) << e_vis_y(0), e_vis_y(1), e_vis_y(2);
    src_points.col(j+3) << e_vis_z(0), e_vis_z(1), e_vis_z(2);
  }


  // Estimates the aligning transformation from camera to gnss coordinate system
  const bool estimate_scale = true;
  Eigen::Matrix4d Tgw_mat = Eigen::umeyama(src_points, dst_points, estimate_scale);
  const auto Tgw = Sophus::Sim3d(Tgw_mat);
  return Tgw;
}
