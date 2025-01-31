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

  const float FEATURE_SIZE = 9.0;

  class GpuFast {
    short2 * kpLoc;
    float * kpScore;
    unsigned int * counter_ptr;
    unsigned int highThreshold;
    unsigned int lowThreshold; 
    unsigned int maxKeypoints;
    unsigned int count;
    cv::cuda::GpuMat scoreMat;
    cudaStream_t stream;
    Stream cvStream;
  public:
    GpuFast(int highThreshold, int lowThreshold, int maxKeypoints = 10000);
    ~GpuFast();

    // void detect(InputArray, std::vector<KeyPoint>&);

    // void detectAsync(InputArray);
    // void joinDetectAsync(std::vector<KeyPoint>&);
    void detectAsyncOpenCv(InputArray);
  };
}
#endif