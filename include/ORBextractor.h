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
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <cuda/Fast.hpp>
#include <cuda/Orb.hpp>
#include <cuda/Angle.hpp>
#include <KeyPoint.h>

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core/cuda.hpp>

namespace ORB_SLAM3
{

    class ExtractorNode
    {
    public:
        ExtractorNode() : bNoMore(false) {}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<ORB_SLAM3::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class ORBextractor
    {
    public:
        ORBextractor(int nFeatures,int nFastFeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST, int imageWidth, int imageHeight);

        ~ORBextractor() {}

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        std::optional<std::tuple<std::shared_ptr<std::vector<KeyPoint>>, cv::cuda::HostMem>> extractFeatures(const cv::cuda::HostMem &im_managed);

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
        std::optional<std::tuple<std::vector<size_t>,std::vector<size_t>,cuda::managed::ManagedVector<KeyPoint>::SharedPtr>> ComputeKeyPointsOctTree();
        
        std::list<ExtractorNode> DistributeOctTree(const unsigned int fastKpCount, const short2 * location, const int* response, const int minX,
                                                    const int maxX, const int minY, const int maxY, const int maxFeatures, const int level);

        
        constexpr static float factorPI = (float)(CV_PI/180.f);
        int nfeatures;
        int nFastFeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        cv::Ptr<cv::cuda::Filter> gpuGaussian;

        std::unique_ptr<std::vector<short2>> kpLoc;
        std::unique_ptr<std::vector<int>> response;

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
