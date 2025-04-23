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

#pragma once

#include <vector>
#include <list>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <cuda/Fast.hpp>
#include <cuda/Orb.hpp>
#include <cuda/Angle.hpp>
#include <cuda/ManagedVector.hpp>

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>

namespace ORB_SLAM3
{

    class ExtractorNode
    {
    public:
        ExtractorNode() : bNoMore(false) {}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<ORB_SLAM3::cuda::managed::KeyPoint> vKeys;
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

        ORBextractor(int nFeatures,int nFastFeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST, int imageWidth, int imageHeight);

        ~ORBextractor() {}

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        int extractFeatures(const cv::cuda::HostMem &im_managed,
                       std::shared_ptr<std::vector<cv::KeyPoint>> &_keypoints,
                       cv::cuda::HostMem& _descriptors);

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

        std::vector<cv::cuda::HostMem> mvImagePyramid;
        std::vector<cv::cuda::HostMem> mvBlurredImagePyramid;

    protected:
        void AllocatePyramid(int width, int height);
        void ComputePyramid(cv::cuda::HostMem image_managed);
        std::tuple<int,std::vector<cuda::managed::ManagedVector::SharedPtr>> ComputeKeyPointsOctTree();
        void computeDescriptors(cv::cuda::HostMem image_managed, std::vector<cv::KeyPoint>& keypointsLevel, std::vector<cv::KeyPoint>& keypointsTotal, cv::Mat& descriptors,
                                   const std::vector<cv::Point>& pattern, int monoIndexOffset, float scaleFactor, int level);
        
        cuda::managed::ManagedVector::SharedPtr DistributeOctTree(const unsigned int fastKpCount, const short2 * location, const int* response, const int minX,
                                                    const int maxX, const int minY, const int maxY, const int maxFeatures, const int level);

        
        constexpr static float factorPI = (float)(CV_PI/180.f);
        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        cv::Ptr<cv::cuda::Filter> gpuGaussian;

        std::vector<int> mnFeaturesPerLevel;
        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        cuda::orb::GpuOrb gpuOrb;
        cuda::fast::GpuFast gpuFast;
        cuda::angle::Angle gpuAngle;
    };

} // namespace ORB_SLAM
