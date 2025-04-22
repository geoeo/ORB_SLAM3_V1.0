#pragma once

#include <opencv2/core/cuda.hpp>
#include "cuda/ManagedVector.hpp"
#include <cuda_runtime.h>

namespace ORB_SLAM3::cuda::angle {
    class Angle {
        cudaStream_t stream;
      public:
        Angle();
        ~Angle();
        void launch_async(cv::cuda::GpuMat image, ORB_SLAM3::cuda::managed::KeyPoint * keypoints, int npoints, int half_k);

        cudaStream_t getStream() { return stream;}
        static void loadUMax(const int* u_max, int count);
      };
}