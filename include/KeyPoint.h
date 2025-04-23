#pragma once

#include <opencv2/core/types.hpp>

namespace ORB_SLAM3
{
  struct KeyPoint {
    cv::Point2f pt;
    float response;
    float size;
    int octave;
    float angle;

    KeyPoint(int x_in, int y_in, int size_in, int response_in=0, int octave_in=0, float angle_in=-1)
        : pt(x_in,y_in), response(response_in), size(size_in), octave(octave_in), angle(angle_in) {
    }
    KeyPoint() = default;
  };
}