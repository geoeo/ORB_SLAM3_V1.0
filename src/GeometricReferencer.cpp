#include <GeometricReferencer.hpp>
#include <Verbose.h>

using namespace std;
using namespace ORB_SLAM3;

GeometricReferencer::GeometricReferencer(int min_nrof_frames)
: m_is_initialized(false),
  m_min_nrof_frames(min_nrof_frames),
  mTgw_current(Sophus::Sim3d())
{
}

void GeometricReferencer::clear()
{
  m_is_initialized = false;
  mTgw_current = Sophus::Sim3d();
  m_latest_frames_to_georef = {};
}

void GeometricReferencer::clearFrames()
{
  m_latest_frames_to_georef = {};
}

bool GeometricReferencer::isInitialized() const
{
  return m_is_initialized;
}

Sophus::Sim3d GeometricReferencer::getCurrentTransform() const 
{
   return mTgw_current; 
}

void GeometricReferencer::addKeyFrame(KeyFrame* kf){
    if(m_latest_frames_to_georef.size() >= m_min_nrof_frames)
      m_latest_frames_to_georef.pop_front();
    m_latest_frames_to_georef.push_back(kf);
}

std::deque<KeyFrame*> GeometricReferencer::getFramesToGeoref() {
  return m_latest_frames_to_georef;
}

optional<Sophus::Sim3d> GeometricReferencer::init(const std::deque<KeyFrame*> &frames)
{

  if (m_is_initialized)
    return mTgw_current;
    
  if (frames.size() < m_min_nrof_frames)
    return nullopt;

  auto pose = update(m_latest_frames_to_georef); 

  m_is_initialized = true;
  return pose;
}

Sophus::Sim3d GeometricReferencer::update(const std::deque<KeyFrame *> &spatials)
{ 
  const auto pose = estimateGeorefTransform(spatials, true);
  mTgw_current = pose*mTgw_current; //TODO: Fix this

  Verbose::PrintMess("FULL Georef function successful", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("Transformation matrix:", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(mTgw_current.rotationMatrix()(0,0)) + " " + to_string(mTgw_current.rotationMatrix()(0,1)) + " " + to_string(mTgw_current.rotationMatrix()(0,2)) + " " + to_string(mTgw_current.translation()(0)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(mTgw_current.rotationMatrix()(1,0)) + " " + to_string(mTgw_current.rotationMatrix()(1,1)) + " " + to_string(mTgw_current.rotationMatrix()(1,2)) + " " + to_string(mTgw_current.translation()(1)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(mTgw_current.rotationMatrix()(2,0)) + " " + to_string(mTgw_current.rotationMatrix()(2,1)) + " " + to_string(mTgw_current.rotationMatrix()(2,2)) + " " + to_string(mTgw_current.translation()(2)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("Scale: " + to_string(mTgw_current.scale()), Verbose::VERBOSITY_NORMAL);

  return mTgw_current;
}

Sophus::Sim3d GeometricReferencer::estimateGeorefTransform(const std::deque<KeyFrame *> &spatials, bool estimate_scale)
{
  // First define basic eigen variables
  const auto measurements = 4;
  auto nrof_points = spatials.size()*measurements;
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> src_points(3, nrof_points);
  Eigen::Matrix< Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> dst_points(3, nrof_points);

  //const auto src_offset = mTgw_current*spatials[0]->GetPoseInverse().translation().cast<double>();
  const auto src_offset = Sophus::Sim3d().translation();

  Verbose::PrintMess("Src Offset: " + to_string(src_offset(0)) + " " + to_string(src_offset(1)) + " " + to_string(src_offset(2)), Verbose::VERBOSITY_NORMAL);

  int i, j;
  for (i = 0, j = 0; i < spatials.size(); ++i, j+=measurements)
  {
    const auto f = spatials[i];
    // The measurements of the gnss receiver
    auto T_rec2g_gis = f->GetGNSSCameraPose();

    auto e_gis_0 = (T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));
    auto e_gis_x = T_rec2g_gis* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    auto e_gis_y = T_rec2g_gis* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    auto e_gis_z = T_rec2g_gis* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    dst_points.col(j) << e_gis_0(0), e_gis_0(1), e_gis_0(2);
    dst_points.col(j+1) << e_gis_x(0), e_gis_x(1), e_gis_x(2);
    dst_points.col(j+2) << e_gis_y(0), e_gis_y(1), e_gis_y(2);
    dst_points.col(j+3) << e_gis_z(0), e_gis_z(1), e_gis_z(2);

    dst_points.col(j) -= src_offset;
    dst_points.col(j+1) -= src_offset;
    dst_points.col(j+2) -= src_offset;
    dst_points.col(j+3) -= src_offset;

    const auto Twc = f->GetPoseInverse();
    const auto Twc_sim3 = Sophus::Sim3d(1.0,Twc.unit_quaternion().cast<double>(), Twc.translation().cast<double>());
    const auto T_c2g = mTgw_current*Twc_sim3;

    auto e_vis_0 = T_c2g* Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    auto e_vis_x = T_c2g* Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
    auto e_vis_y = T_c2g* Eigen::Vector4d(0.0, 1.0, 0.0, 1.0);
    auto e_vis_z = T_c2g* Eigen::Vector4d(0.0, 0.0, 1.0, 1.0);

    src_points.col(j ) << e_vis_0(0), e_vis_0(1), e_vis_0(2);
    src_points.col(j+1) << e_vis_x(0), e_vis_x(1), e_vis_x(2);
    src_points.col(j+2) << e_vis_y(0), e_vis_y(1), e_vis_y(2);
    src_points.col(j+3) << e_vis_z(0), e_vis_z(1), e_vis_z(2);

    src_points.col(j) -= src_offset;
    src_points.col(j+1) -= src_offset;  
    src_points.col(j+2) -= src_offset;
    src_points.col(j+3) -= src_offset;

  }

  // for(int c = 0; c < spatials.size(); ++c){
  //   Verbose::PrintMess("Src: " + to_string(src_points(0,c)) + " " + to_string(src_points(1,c)) + " " + to_string(src_points(2,c)), Verbose::VERBOSITY_NORMAL);
  //   Verbose::PrintMess("Dst: " + to_string(dst_points(0,c)) + " " + to_string(dst_points(1,c)) + " " + to_string(dst_points(2,c)), Verbose::VERBOSITY_NORMAL);
  // }


  // Estimates the aligning transformation from camera to gnss coordinate system
  Eigen::Matrix4d Tgw_mat = Eigen::umeyama(src_points, dst_points, estimate_scale);
  const auto Tgw = Sophus::Sim3d(Tgw_mat);

  Verbose::PrintMess("Georef function successful", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("Transformation matrix:", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(Tgw.rotationMatrix()(0,0)) + " " + to_string(Tgw.rotationMatrix()(0,1)) + " " + to_string(Tgw.rotationMatrix()(0,2)) + " " + to_string(Tgw.translation()(0)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(Tgw.rotationMatrix()(1,0)) + " " + to_string(Tgw.rotationMatrix()(1,1)) + " " + to_string(Tgw.rotationMatrix()(1,2)) + " " + to_string(Tgw.translation()(1)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess(to_string(Tgw.rotationMatrix()(2,0)) + " " + to_string(Tgw.rotationMatrix()(2,1)) + " " + to_string(Tgw.rotationMatrix()(2,2)) + " " + to_string(Tgw.translation()(2)), Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("Scale: " + to_string(Tgw.scale()), Verbose::VERBOSITY_NORMAL);

  return Tgw;
}
