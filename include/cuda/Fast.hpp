#pragma once

#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM3::cuda::fast {
  using namespace std;
  using namespace cv;
  using namespace cv::cuda;

  class GpuFast {
    short2 * kpLoc;
    short2 * kpLocFinal;
    float * kpResponse;
    float * kpResponseFinal;
    unsigned int * counter_ptr;
    unsigned int maxKeypoints;
    int imHeight;
    int imWidth;

    int* scoreMat;
    cudaStream_t stream;
  public:
    GpuFast(int imHeight, int imWidth, int maxKeypoints);
    ~GpuFast();
    void detect(const cv::cuda::GpuMat image, int threshold, int borderX, int borderY, std::vector<KeyPoint>& keypoints);
    cudaStream_t getStream();
  };
}