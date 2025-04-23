#pragma once

namespace ORB_SLAM3
{
  struct KeyPoint {
    int x;
    int y;
    int response;
    int size;
    int octave;
    float angle;

    KeyPoint(short x_in, short y_in, int response_in, int size_in, int octave_in, float angle_in)
        : x(x_in), y(y_in), response(response_in), size(size_in), octave(octave_in), angle(angle_in) {
    }
  };
}