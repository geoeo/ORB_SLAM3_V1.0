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

        void prefetchToCPU(cudaStream_t stream = 0) {
            // Not supported on Orin NX - https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#cuda-features-not-supported-on-tegra
            checkCudaErrors( cudaMemPrefetchAsync(unified_ptr_, sizeInBytes_, cudaCpuDeviceId, stream));
            checkCudaErrors( cudaStreamSynchronize(stream) );
            prefetchCPU_ = true;
        }

        
        T * getHostPtr(cudaStream_t stream = 0, size_t offset = 0) {
            if(!prefetchCPU_){
                checkCudaErrors( cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachHost) );
                checkCudaErrors( cudaStreamSynchronize(stream) ); //TODO: check if this is necessary  - maybe use prefetch async
                prefetchCPU_ = true;
            }
            return unified_ptr_ + offset;
        }

        T * getDevicePtr(cudaStream_t stream = 0, size_t offset = 0) {
            prefetchCPU_ = false;
            checkCudaErrors( cudaStreamAttachMemAsync(stream, unified_ptr_, 0, cudaMemAttachGlobal) );
            checkCudaErrors( cudaStreamSynchronize(stream) ); //TODO: check if this is necessary - maybe use prefetch async
            return unified_ptr_ + offset;
        }

        std::shared_ptr<std::vector<T>> toVector() {
            auto vec = std::make_shared<std::vector<T>>(size());
            auto hostPtr = getHostPtr();
            std::memcpy(vec->data(), hostPtr, sizeInBytes_);
            return vec;
        }
        
        size_t size() const {
            return  sizeInBytes_ / sizeof(T);
        }

        size_t sizeInBytes() const {
            return sizeInBytes_;
        }
        
        private:
            // Using unified memory
            T *unified_ptr_;
            size_t sizeInBytes_;
            bool prefetchCPU_;

            ManagedVector(size_t sizeInBytes) : sizeInBytes_(sizeInBytes), prefetchCPU_(false) {
                checkCudaErrors(cudaMallocManaged(&unified_ptr_, sizeInBytes));
            }

            ~ManagedVector() {
                checkCudaErrors(cudaFree(unified_ptr_));
            };
    };

}