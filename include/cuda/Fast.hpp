#pragma once

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

namespace ORB_SLAM3::cuda::fast {
  class GpuFast {
    short2 * kpLoc;
    short2 * kpLocFinal;
    int * kpResponseFinal;
    unsigned int * counter_ptr;
    unsigned int maxKeypoints;
    int imHeight;
    int imWidth;

    int* scoreMat;
    cudaStream_t stream;
  public:
    GpuFast(int imHeight, int imWidth, int maxKeypoints);
    ~GpuFast();
    unsigned int detect(const cv::cuda::GpuMat image, int threshold, int borderX, int borderY);
    cudaStream_t getStream();
    short2 * getLoc();
    int* getResp();
  };
}