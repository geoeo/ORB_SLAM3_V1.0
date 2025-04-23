#pragma once

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>
#include "cuda/HelperCuda.h"
#include <iostream>

#define checkCudaErrors(val) ORB_SLAM3::cuda::CUDAHelper::check((val), #val, __FILE__, __LINE__)

namespace ORB_SLAM3::cuda::managed
{

    struct KeyPoint {
        int x;
        int y;
        int response;
        int size;
        int octave;
        float angle;
        int _pad1;
        int _pad2;

        KeyPoint(short x_in, short y_in, int response_in, int size_in, int octave_in, float angle_in)
            : x(x_in), y(y_in), response(response_in), size(size_in), octave(octave_in), angle(angle_in) {
        }
    };

    /**
     * This struct wrapped a CUDA unified memory ptr. 
     * It should only be used wrapped in a SharedPtr.
     * Therefore, Default and Copy Constructors are disabled.
     */
    struct ManagedVector {
        using SharedPtr = std::shared_ptr<ManagedVector>;

        struct ManagedVectorDeleter
        {
            void operator()(ManagedVector* p) const { delete p; }
        };

        ManagedVector() = delete;
        ManagedVector(const ManagedVector&) = delete;
        ManagedVector & operator=(const ManagedVector& other) = delete;
        
        static ManagedVector::SharedPtr CreateManagedVector(size_t numberOfKeypoints){
            std::cout << "Num: " << numberOfKeypoints << std::endl;
            std::cout << "Size: " << sizeof(KeyPoint) << std::endl;
            const auto sizeInBytes = numberOfKeypoints*sizeof(KeyPoint);
            return std::shared_ptr<ManagedVector>(new ManagedVector(sizeInBytes),ManagedVectorDeleter{}); 
        }

        /**
         * This function exposes the raw ptr. 
         * Make sure the lifetime of the bound datastructures are less than the CUDAManagedMemory struct.
         */

        KeyPoint * getHostPtr(cudaStream_t stream) {

            // cudaDeviceSynchronize();
            // checkCudaErrors( cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachHost) );
            // checkCudaErrors( cudaStreamSynchronize(stream) );
            // cudaDeviceSynchronize();
            return unified_ptr_;
        }

        KeyPoint * getDevicePtr(cudaStream_t stream) {
            // cudaDeviceSynchronize();
            // checkCudaErrors( cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachGlobal) );
            // checkCudaErrors( cudaStreamSynchronize(stream) );
            // cudaDeviceSynchronize();
            return unified_ptr_;
        }

        size_t getSize() const {
            return  size_in_bytes_ / sizeof(KeyPoint);
        }

        size_t sizeInBytes() const {
            return size_in_bytes_;
        }
        
        KeyPoint& at(unsigned int i, cudaStream_t stream ){ 
            return getHostPtr(stream)[i];
        }

        private:
            // Using unified memory
            KeyPoint *unified_ptr_;
            size_t size_in_bytes_;

            ManagedVector(size_t sizeInBytes) : size_in_bytes_(sizeInBytes) {
                checkCudaErrors(cudaMallocManaged(&unified_ptr_, sizeInBytes));
                cudaDeviceSynchronize();
            }

            ~ManagedVector() {
                checkCudaErrors(cudaFree(unified_ptr_));
                cudaDeviceSynchronize();
            };
    };

}