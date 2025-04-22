#pragma once

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>

namespace ORB_SLAM3::cuda::orb {

  class GpuOrb {
    unsigned int maxKeypoints;
    //cv::KeyPoint * keypoints;
    cudaStream_t stream;
    cv::cuda::Stream cvStream;
  public:
    GpuOrb(int maxKeypoints);
    ~GpuOrb();

    void launch_async(cv::cuda::GpuMat image, cv::cuda::GpuMat descriptors,int offset, int offset_end, cv::KeyPoint * keypoints, const int npoints);
    cudaStream_t getStream();
    cv::cuda::Stream getCvStream();

    static void loadPattern(const cv::Point * _pattern);
  };
} 
