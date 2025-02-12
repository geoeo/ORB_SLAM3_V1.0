#pragma once

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

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
    void detect(const cv::cuda::GpuMat image, int threshold, int borderX, int borderY, std::vector<cv::KeyPoint>& keypoints);
    cudaStream_t getStream();
  };
}