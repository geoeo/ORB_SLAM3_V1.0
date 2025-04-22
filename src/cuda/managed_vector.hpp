#pragma once

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>
#include <opencv2/core/types.hpp>

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
        
        static ManagedVector::SharedPtr CreateManagedVector(size_t sizeInBytes){
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
                checkCudaError(cudaMallocManaged(&unified_ptr_, sizeInBytes), __FILE__, __LINE__);
            }

            ~ManagedVector() {
                checkCudaError(cudaFree(unified_ptr_), __FILE__, __LINE__);
            };

            static void checkCudaError(cudaError_t result, const char *const file, int const line){
                if(result != cudaError_t::cudaSuccess){
                    std::stringstream ss;
                    ss << "CUDA error at " << file <<":"<<line << " "<< cudaGetErrorString(result) << std::endl;
                    throw std::runtime_error(ss.str());
                }
            }
    };

}