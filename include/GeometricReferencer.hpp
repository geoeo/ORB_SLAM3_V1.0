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
    std::deque<KeyFrame*> getFramesToGeoref();
    std::optional<Sophus::Sim3d> apply(const std::deque<KeyFrame*> &frames, bool do_update);
    Sophus::Sim3d update(const std::deque<KeyFrame *> &spatials);
    Sophus::Sim3d getCurrentTransform() const;

    bool isInitialized() const;
    void clear();
    void clearFrames();

  private:
    bool m_is_initialized;
    int m_min_nrof_frames;
    Sophus::Sim3d mTgw_current;
    std::deque<KeyFrame*> m_latest_frames_to_georef;
    std::mutex mMutexFrames;

    Sophus::Sim3d estimateGeorefTransform(const std::deque<KeyFrame *> &spatials, bool estimate_scale);
  };

} // namespace ORB_SLAM3