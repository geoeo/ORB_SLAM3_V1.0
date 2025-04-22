#pragma once

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>
#include <opencv2/core/types.hpp>
#include "cuda/HelperCuda.h"

#define checkCudaErrors(val) ORB_SLAM3::cuda::CUDAHelper::check((val), #val, __FILE__, __LINE__)

namespace ORB_SLAM3::cuda::managed
{

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
            const auto sizeInBytes = numberOfKeypoints*sizeof(cv::KeyPoint);
            return std::shared_ptr<ManagedVector>(new ManagedVector(sizeInBytes),ManagedVectorDeleter{}); 
        }

        /**
         * This function exposes the raw ptr. 
         * Make sure the lifetime of the bound datastructures are less than the CUDAManagedMemory struct.
         */
        void * getRawPtr() {
            return unified_ptr_;
        }

        cv::KeyPoint * getPtr() {
            return (cv::KeyPoint *) unified_ptr_;
        }

        uint32_t getSize() const {
            return  size_in_bytes_ / sizeof(cv::KeyPoint);
        }

        uint32_t sizeInBytes() const {
            return size_in_bytes_;
        }
        
        cv::KeyPoint& operator[](unsigned int i){ 
            return getPtr()[i];
        }

        private:
            // Using unified memory
            void *unified_ptr_;
            uint32_t size_in_bytes_;

            ManagedVector(size_t sizeInBytes) : size_in_bytes_(sizeInBytes) {
                checkCudaErrors(cudaMallocManaged(&unified_ptr_, sizeInBytes));
            }

            ~ManagedVector() {
                checkCudaErrors(cudaFree(unified_ptr_));
            };
    };

}