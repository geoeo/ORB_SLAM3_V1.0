#pragma once 

#include <memory>
#include <numeric>
#include <queue>
#include <deque>
#include <mutex>
#include <utility>
#include <optional>

#include <KeyFrame.h>

namespace ORB_SLAM3
{

  class GeometricReferencer
  {
  public:
    explicit GeometricReferencer(int min_nrof_frames);

    void addKeyFrame(KeyFrame* kf);
    std::vector<KeyFrame*> getFramesToGeoref();
    std::optional<std::pair<Sophus::SE3d, double>> init(const std::vector<KeyFrame*> &frames);
    std::pair<Sophus::SE3d, double> update(const std::vector<KeyFrame*> &frames);
    std::pair<Sophus::SE3d, double> getCurrentTransform() const;

    bool isInitialized() const;
    void clear();
    void clearFrames();

  private:
    bool m_is_initialized;
    int m_min_nrof_frames;
    Sophus::SE3d mTgw_current;
    double mSgw_current;

    std::vector<KeyFrame*> m_frames_to_georef;
    std::deque<std::pair<Sophus::SE3d, Sophus::SE3d>> m_spatials;

    static std::pair<Sophus::SE3d, double> estimateGeorefTransform(const std::deque<std::pair<Sophus::SE3d, Sophus::SE3d>> &spatials, const Sophus::SE3d &T_w2g_init, bool estimate_scale);

  };

} // namespace ORB_SLAM3