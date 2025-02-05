#pragma once


#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cuda_runtime.h>
#include <cuda/Cuda.hpp>

namespace ORB_SLAM3::cuda::fastCV {
  using namespace std;
  using namespace cv;
  using namespace cv::cuda;

  class GpuFastCV {
    short2 * kpLoc;
    float * kpScore;
    unsigned int * counter_ptr;
    int highThreshold;
    int lowThreshold; 
    int imHeight;
    int imWidth;
    int gridSize;
    int maxKeypoints;
    int numGridX;
    int numGridY;
    int gridTileNum;
    unsigned int count;

    static const int LOCATION_ROW = 0;
    static const int RESPONSE_ROW = 1;
    static const int FEATURE_SIZE = 7;

    cv::cuda::GpuMat scoreMat;
    cudaStream_t stream;
    Stream cvStream;
  public:
    GpuFastCV(int highThreshold, int lowThreshold, int imHeight, int imWidth, int gridSize, int maxKeypoints = 10000);
    ~GpuFastCV();
    void detectAsyncOpenCv(InputArray im, std::vector<KeyPoint>& keypoints_cpu);
  };
}