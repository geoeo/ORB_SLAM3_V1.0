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
      public:
        Angle(unsigned int maxKeypoints);
        ~Angle();
        void launch_async(cv::cuda::GpuMat image, cv::KeyPoint * _keypoints, int npoints, int half_k);

        cudaStream_t getStream() { return stream;}
        static void loadUMax(const int* u_max, int count);
      };
}