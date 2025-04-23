#pragma once

#include <opencv2/core/types.hpp>

namespace ORB_SLAM3
{
  struct KeyPoint {
    cv::Point pt;
    int response;
    int size;
    int octave;
    float angle;

    KeyPoint(short x_in, short y_in, int response_in, int size_in, int octave_in, float angle_in)
        : pt(x_in,y_in), response(response_in), size(size_in), octave(octave_in), angle(angle_in) {
    }
  };
}