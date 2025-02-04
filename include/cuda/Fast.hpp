#pragma once

#ifndef __FAST_HPP__
#define __FAST_HPP__

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM3::cuda::fast {
  using namespace std;
  using namespace cv;
  using namespace cv::cuda;

  class GpuFast {
    short2 * kpLoc;
    float * kpScore;
    unsigned int * counter_ptr;
    int highThreshold;
    int lowThreshold; 
    int gridSizeX;
    int gridSizeY;
    int maxKeypoints;
    unsigned int count;

    static const int LOCATION_ROW = 0;
    static const int RESPONSE_ROW = 1;
    static const int FEATURE_SIZE = 7;

    cv::cuda::GpuMat scoreMat;
    cudaStream_t stream;
    Stream cvStream;
  public:
    GpuFast(int highThreshold, int lowThreshold, int grid_size_x, int grid_size_y, int maxKeypoints = 10000);
    ~GpuFast();

    // void detect(InputArray, std::vector<KeyPoint>&);

    // void detectAsync(InputArray);
    // void joinDetectAsync(std::vector<KeyPoint>&);
    void detectAsyncOpenCv(InputArray im, std::vector<KeyPoint>& keypoints_cpu);
  };
}
#endif