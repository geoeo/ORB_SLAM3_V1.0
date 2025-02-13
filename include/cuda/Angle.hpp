#pragma once

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM3::cuda::angle {
    class Angle {
        unsigned int maxKeypoints;
        cv::KeyPoint * keypoints;
        cudaStream_t stream;
        cv::cuda::Stream _cvStream;
      public:
        Angle(unsigned int maxKeypoints = 10000);
        ~Angle();
        void launch_async(cv::cuda::GpuMat image, cv::KeyPoint * _keypoints, int npoints, int half_k);

        cudaStream_t getStream() { return stream;}
        cv::cuda::Stream& cvStream() { return _cvStream;}
        static void loadUMax(const int* u_max, int count);
      };
}