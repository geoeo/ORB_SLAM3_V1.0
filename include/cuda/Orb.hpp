#pragma once

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM3::cuda::orb {

  class GpuOrb {
    unsigned int maxKeypoints;
    cv::KeyPoint * keypoints;
    cv::cuda::GpuMat descriptors;
    cv::cuda::GpuMat desc;
    cudaStream_t stream;
    cv::cuda::Stream cvStream;
  public:
    GpuOrb(int maxKeypoints);
    ~GpuOrb();

    void launch_async(cv::cuda::GpuMat image, const cv::KeyPoint * _keypoints, const int npoints);
    void join(cv::Mat &_descriptors);
    cudaStream_t getStream();

    static void loadPattern(const cv::Point * _pattern);
  };
} 
