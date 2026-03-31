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
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Sophus/sophus/sim3.hpp>

#include <MapPoint.h>
#include <KeyFrame.h>
#include <Frame.h>


namespace ORB_SLAM3
{

    class ORBmatcher
    {
    public:
        // Computes the Hamming distance between two ORB descriptors
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
        // Used to track the local map (Tracking)
        static int SearchByProjection(std::shared_ptr<Frame> F, const std::vector<std::shared_ptr<MapPoint>> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints, const float nnRatio, const bool checkOrientation);

        // Project MapPoints seen in KeyFrame into the Frame and search matches.
        // Used in relocalisation (Tracking)
        static int SearchByProjection(std::shared_ptr<Frame> CurrentFrame, std::shared_ptr<KeyFrame> pKF, const std::set<std::shared_ptr<MapPoint>> &sAlreadyFound, const float th, const bool checkOrientation);

        // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
        // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
        // Used in Relocalisation and Loop Detection
        static int SearchByBoW(std::shared_ptr<KeyFrame> pKF, std::shared_ptr<Frame> F, std::vector<std::shared_ptr<MapPoint>> &vpMapPointMatches, const float nnRatio, const bool checkOrientation);

        // Matching for the Map Initialization (only used in the monocular case)
        static std::pair<int,std::vector<int>> SearchForInitialization(std::shared_ptr<Frame> F1, std::shared_ptr<Frame> F2, int windowSize, const float nnRatio, const bool checkOrientation);

        // Matching to triangulate new MapPoints. Check Epipolar Constraint.
        static int SearchForTriangulation(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2,
                                   std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse, const bool checkOrientation);

        // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
        // In the stereo and RGB-D case, s12=1
        static int SearchBySim3(std::shared_ptr<KeyFrame> pKF1, std::shared_ptr<KeyFrame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches12, const Sophus::Sim3f &S12, const float th);

        // Project MapPoints into KeyFrame and search for duplicated MapPoints.
        static int Fuse(std::shared_ptr<KeyFrame> pKF, const vector<std::shared_ptr<MapPoint>> &vpMapPoints, const float th=3.0, const bool bRight = false);

        // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
        static int Fuse(std::shared_ptr<KeyFrame> pKF, Sophus::Sim3f &Scw, const std::vector<std::shared_ptr<MapPoint>> &vpPoints, float th, std::vector<std::shared_ptr<MapPoint>> &vpReplacePoint);

    public:
        static constexpr int TH_LOW = 30;
        static constexpr int TH_HIGH = 100;
        static constexpr size_t HISTO_LENGTH = 30;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        static float RadiusByViewingCos(const float &viewCos);

        static void ComputeThreeMaxima(std::array<std::vector<int>, HISTO_LENGTH> &histo, const int L, int &ind1, int &ind2, int &ind3);
    };

}// namespace ORB_SLAM