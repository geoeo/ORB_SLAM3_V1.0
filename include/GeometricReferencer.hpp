#pragma once 

#include <memory>
#include <numeric>
#include <deque>
#include <mutex>
#include <utility>

#include <opencv2/core/core.hpp>
#include <KeyFrame.h>

namespace ORB_SLAM3
{

  class GeometricReferencer
  {
  public:
    explicit GeometricReferencer(double th_error, int min_nrof_frames);

    bool init(const std::deque<KeyFrame*> &frames);
    void update(const std::deque<KeyFrame*> &frames);

    double getScale();
    cv::Mat getTransform();
    bool isInitialized();
    void clear();

  private:
    bool m_is_initialized;

    cv::Mat m_T_w2g;
    double m_scale;

    size_t m_prev_nrof_unique;

    double m_th_error;
    double m_error;

    int m_min_nrof_frames;

    std::mutex m_mutex_transform;

    std::deque<std::pair<cv::Mat, cv::Mat>> m_spatials;

    static cv::Mat estimateGeorefTransform(const std::deque<std::pair<cv::Mat, cv::Mat>> &spatials, const cv::Mat &T_w2g_init, bool estimate_scale);

  };

} // namespace realm