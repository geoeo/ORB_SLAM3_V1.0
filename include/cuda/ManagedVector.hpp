#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda/HelperCuda.h>

#define checkCudaErrors(val) ORB_SLAM3::cuda::CUDAHelper::check((val), #val, __FILE__, __LINE__)

namespace ORB_SLAM3::cuda::managed
{


    /**
     * This struct wrapped a CUDA unified memory ptr. 
     * It should only be used wrapped in a SharedPtr.
     * Therefore, Default and Copy Constructors are disabled.
     */
    template <typename T>
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
            const auto sizeInBytes = numberOfKeypoints*sizeof(T);
            return std::shared_ptr<ManagedVector>(new ManagedVector(sizeInBytes),ManagedVectorDeleter{}); 
        }

        
        T * getHostPtr(cudaStream_t stream = 0) {
            checkCudaErrors( cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachHost) );
            checkCudaErrors( cudaStreamSynchronize(stream) );
            return unified_ptr_;
        }

        T * getDevicePtr(cudaStream_t stream = 0) {
            checkCudaErrors( cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachSingle) );
            checkCudaErrors( cudaStreamSynchronize(stream) );
            return unified_ptr_;
        }

        std::shared_ptr<std::vector<T>> toVector() {
            auto vec = std::make_shared<std::vector<T>>(size());
            auto hostPtr = getHostPtr();
            std::memcpy(vec->data(), hostPtr, size_in_bytes_);
            return vec;
        }
        
        size_t size() const {
            return  size_in_bytes_ / sizeof(T);
        }

        size_t sizeInBytes() const {
            return size_in_bytes_;
        }
        
        private:
            // Using unified memory
            T *unified_ptr_;
            size_t size_in_bytes_;

            ManagedVector(size_t sizeInBytes) : size_in_bytes_(sizeInBytes) {
                checkCudaErrors(cudaMallocManaged(&unified_ptr_, sizeInBytes));
            }

            ~ManagedVector() {
                checkCudaErrors(cudaFree(unified_ptr_));
            };
    };

}