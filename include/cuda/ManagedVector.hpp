#pragma once

#include <cstddef>
#include <memory>
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
        
        size_t getSize() const {
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