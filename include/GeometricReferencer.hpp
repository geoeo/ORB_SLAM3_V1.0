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
    std::deque<KeyFrame*> getFramesForGeorefEstimation();
    std::vector<KeyFrame*> getFramesWithoutGeoref();
    std::optional<Sophus::Sim3d> apply(const std::deque<KeyFrame*> &frames, bool do_update);
    Sophus::Sim3d update(const std::deque<KeyFrame *> &spatials);
    Sophus::Sim3d getCurrentTransform();

    bool isInitialized() const;
    void clear();
    void clearFrames();
    void updateGeorefKFsCount(size_t count);
  private:
    bool m_is_initialized;
    int m_min_nrof_frames;
    size_t m_georefed_kfs_count;
    Sophus::Sim3d mTgw_current;
    std::deque<KeyFrame*> m_latest_frames_to_georef;
    std::mutex mMutexFrames;
    std::mutex mMutexTransform;

    Sophus::Sim3d estimateGeorefTransform(const std::deque<KeyFrame *> &spatials);
  };

} // namespace ORB_SLAM3