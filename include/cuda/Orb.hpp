#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "cuda/ManagedVector.hpp"
#include <cuda_runtime.h>

namespace ORB_SLAM3::cuda::orb {

  class GpuOrb {
    cudaStream_t stream;
    cv::cuda::Stream cvStream;
  public:
    GpuOrb();
    ~GpuOrb();

    void launch_async(cv::cuda::GpuMat image, cv::cuda::GpuMat descriptors,int offset, int offset_end, ORB_SLAM3::cuda::managed::KeyPoint * keypoints, const int npoints, float scale);
    cudaStream_t getStream();
    cv::cuda::Stream getCvStream();

    static void loadPattern(const cv::Point * _pattern);
  };
} 
