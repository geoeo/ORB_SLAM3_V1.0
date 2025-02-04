/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <cuda/Fast.hpp>

#include <CUDACvManagedMemory/cuda_cv_managed_memory.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>

namespace ORB_SLAM3
{

    class ExtractorNode
    {
    public:
        ExtractorNode() : bNoMore(false) {}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class ORBextractor
    {
    public:
        enum
        {
            HARRIS_SCORE = 0,
            FAST_SCORE = 1
        };

        ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST, int gridCount,
                     int imageWidth, int imageHeight);

        ~ORBextractor() {}

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        int operator()(const cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr &im_managed, cv::InputArray _mask,
                       std::vector<cv::KeyPoint> &_keypoints,
                       cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

        int inline GetLevels()
        {
            return nlevels;
        }

        float inline GetScaleFactor()
        {
            return scaleFactor;
        }

        std::vector<float> inline GetScaleFactors()
        {
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors()
        {
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares()
        {
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares()
        {
            return mvInvLevelSigma2;
        }

        std::vector<cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr> mvImagePyramid;
        std::vector<cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr> mvBlurredImagePyramid;

    protected:
        void AllocatePyramid(int width, int height);
        void ComputePyramid(cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr image_managed);
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
        void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypointsLevel, std::vector<cv::KeyPoint>& keypointsTotal, cv::Mat& descriptors,
                                   const std::vector<cv::Point>& pattern, int monoIndexOffset, float scaleFactor);
        
        static void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);
        static float IC_Angle(const cv::Mat& image, cv::Point2f pt,  const std::vector<int> & u_max);
        static int getOrbValue(const uchar* center, const cv::Point* pattern, int idx, float a, float b, int step);
        static void computeOrbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern,uchar* desc);
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                    const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        std::vector<cv::Point> pattern;
        
        constexpr static float factorPI = (float)(CV_PI/180.f);
        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;
        float gridCount;

        std::vector<int> mnFeaturesPerLevel;
        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        cuda::fast::GpuFast gpuFast;
        cv::Ptr<cv::Feature2D> feat;
        cv::Ptr<cv::Feature2D> feat_back;
        cv::Ptr<cv::Feature2D> feat_back_gpu;
    };

} // namespace ORB_SLAM

#endif
