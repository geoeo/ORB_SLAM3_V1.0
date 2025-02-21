

#include "cuda/helper_cuda.h"
#include <cuda/Angle.hpp>
#include <algorithm>
#include <iostream>
#include <opencv2/core/cuda_types.hpp>

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/reduce.hpp>
#include <opencv2/core/cuda/functional.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device;

#define checkCudaErrors(val) ORB_SLAM3::cuda::CUDAHelper::check((val), #val, __FILE__, __LINE__)

namespace ORB_SLAM3::cuda::angle {

    __constant__ int c_u_max[32];

    void Angle::loadUMax(const int* u_max, int count)
    {
        checkCudaErrors( cudaMemcpyToSymbol(c_u_max, u_max, count * sizeof(int)));
    }

    __global__ void IC_Angle_kernel(const PtrStepb image, KeyPoint * keypoints, const int npoints, const int half_k)
    {
        __shared__ int smem0[8 * 32];
        __shared__ int smem1[8 * 32];

        int* srow0 = smem0 + threadIdx.y * blockDim.x;
        int* srow1 = smem1 + threadIdx.y * blockDim.x;

        cv::cuda::device::plus<int> op;

        const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

        if (ptidx < npoints) {
            int m_01 = 0, m_10 = 0;

            const short2 loc = make_short2(keypoints[ptidx].pt.x, keypoints[ptidx].pt.y);

            // Treat the center line differently, v=0
            for (int u = threadIdx.x - half_k; u <= half_k; u += blockDim.x)
                m_10 += u * image(loc.y, loc.x + u);

            reduce<32>(srow0, m_10, threadIdx.x, op);

            for (int v = 1; v <= half_k; ++v)
            {
                // Proceed over the two lines
                int v_sum = 0;
                int m_sum = 0;
                const int d = c_u_max[v];

                for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
                {
                    int val_plus = image(loc.y + v, loc.x + u);
                    int val_minus = image(loc.y - v, loc.x + u);

                    v_sum += (val_plus - val_minus);
                    m_sum += u * (val_plus + val_minus);
                }

                reduce<32>(smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum), threadIdx.x, thrust::make_tuple(op, op));

                m_10 += m_sum;
                m_01 += v * v_sum;
            }

            if (threadIdx.x == 0)
            {
                float kp_dir = atan2f((float)m_01, (float)m_10);
                kp_dir += (kp_dir < 0) * (2.0f * CV_PI_F);
                kp_dir *= 180.0f / CV_PI_F;

                keypoints[ptidx].angle = kp_dir;
            }
        }
    }

    Angle::Angle(unsigned int maxKeypoints) : maxKeypoints(maxKeypoints) {
        checkCudaErrors( cudaStreamCreate(&stream) );
        checkCudaErrors( cudaMalloc(&keypoints, sizeof(KeyPoint) * maxKeypoints) );
    }

    Angle::~Angle() {
        checkCudaErrors( cudaFree(keypoints) );
        checkCudaErrors( cudaStreamDestroy(stream) );
    }

    void Angle::launch_async(cv::cuda::GpuMat image, KeyPoint * _keypoints, int npoints, int half_k) {
        checkCudaErrors( cudaMemcpyAsync(keypoints, _keypoints, sizeof(KeyPoint) * npoints, cudaMemcpyHostToDevice, stream) );
        dim3 block(32, 8);
        dim3 grid(divUp(npoints, block.y));
        IC_Angle_kernel<<<grid, block, 0, stream>>>(image, keypoints, npoints, half_k);
        checkCudaErrors( cudaGetLastError() );
        checkCudaErrors( cudaMemcpyAsync(_keypoints, keypoints, sizeof(KeyPoint) * npoints, cudaMemcpyDeviceToHost, stream) );
        checkCudaErrors( cudaStreamSynchronize(stream) );
        
    }
} //namespace
