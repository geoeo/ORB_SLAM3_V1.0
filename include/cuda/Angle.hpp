#pragma once

#include <opencv2/core/cuda.hpp>
#include <cuda/ManagedVector.hpp>
#include <KeyPoint.h>
#include <cuda_runtime.h>

namespace ORB_SLAM3::cuda::angle {
    class Angle {
      public:
        void launch_async(cv::cuda::GpuMat image, ORB_SLAM3::KeyPoint * keypoints, int npoints, int half_k, cudaStream_t stream);

        static void loadUMax(const int* u_max, int count);
      };
}