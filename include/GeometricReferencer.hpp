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

    void addKeyFrame(std::shared_ptr<KeyFrame> kf);
    std::deque<std::shared_ptr<KeyFrame>> getFramesForGeorefEstimation();
    std::vector<std::shared_ptr<KeyFrame>> getFramesWithoutGeoref();
    std::optional<Sophus::Sim3d> apply(const std::deque<std::shared_ptr<KeyFrame>> &frames, bool do_update);
    Sophus::Sim3d update(const std::deque<std::shared_ptr<KeyFrame>> &spatials);
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
    std::deque<std::shared_ptr<KeyFrame>> m_latest_frames_to_georef;
    std::mutex mMutexFrames;
    std::mutex mMutexTransform;

    Sophus::Sim3d estimateGeorefTransform(const std::deque<std::shared_ptr<KeyFrame>> &spatials);
  };

} // namespace ORB_SLAM3