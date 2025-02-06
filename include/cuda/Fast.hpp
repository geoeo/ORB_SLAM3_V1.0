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
    float * kpScore;
    unsigned int * counter_ptr;
    unsigned int highThreshold;
    unsigned int lowThreshold; 
    unsigned int maxKeypoints;
    unsigned int count;
    int imHeight;
    int imWidth;

    static const int LOCATION_ROW = 0;
    static const int RESPONSE_ROW = 1;
    static const int FEATURE_SIZE = 7;

    cv::cuda::GpuMat scoreMat;
    cudaStream_t stream;
    Stream cvStream;
  public:
    GpuFast(int highThreshold, int lowThreshold,int imHeight, int imWidth, int maxKeypoints = 10000);
    ~GpuFast();

    void detect(InputArray, std::vector<KeyPoint>&);
    void detectAsync(InputArray);
    void joinDetectAsync(std::vector<KeyPoint>&);
  };

    class IC_Angle {
    unsigned int maxKeypoints;
    KeyPoint * keypoints;
    cudaStream_t stream;
    Stream _cvStream;
  public:
    IC_Angle(unsigned int maxKeypoints = 10000);
    ~IC_Angle();
    void launch_async(InputArray _image, KeyPoint * _keypoints, int npoints, int half_k, int minBorderX, int minBorderY, int octave, int size);
    void join(KeyPoint * _keypoints, int npoints);

    Stream& cvStream() { return _cvStream;}
    static void loadUMax(const int* u_max, int count);
  };
}