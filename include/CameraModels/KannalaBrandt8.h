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

#include <assert.h>
#include <CameraModels/GeometricCamera.h>
#include <TwoViewReconstruction.h>
#include <KeyPoint.h>

namespace ORB_SLAM3 {
    class KannalaBrandt8 : public GeometricCamera {

    public:
        KannalaBrandt8() : precision(1e-6) {
            mvParameters.resize(8);
            mnId=nNextId++;
            mnType = CAM_FISHEYE;
        }
        KannalaBrandt8(const std::vector<float> _vParameters) : GeometricCamera(_vParameters), mvLappingArea(2,0), precision(1e-6) ,tvr(nullptr) {
            assert(mvParameters.size() == 8);
            mnId=nNextId++;
            mnType = CAM_FISHEYE;
        }

        KannalaBrandt8(const std::vector<float> _vParameters, const float _precision) : GeometricCamera(_vParameters),
                                                                                        mvLappingArea(2,0), precision(_precision) {
            assert(mvParameters.size() == 8);
            mnId=nNextId++;
            mnType = CAM_FISHEYE;
        }
        KannalaBrandt8(KannalaBrandt8* pKannala) : GeometricCamera(pKannala->mvParameters), mvLappingArea(2,0), precision(pKannala->precision) ,tvr(nullptr) {
            assert(mvParameters.size() == 8);
            mnId=nNextId++;
            mnType = CAM_FISHEYE;
        }

        cv::Point2f project(const cv::Point3f &p3D);
        Eigen::Vector2d project(const Eigen::Vector3d & v3D);
        Eigen::Vector2f project(const Eigen::Vector3f & v3D);
        Eigen::Vector2f projectMat(const cv::Point3f& p3D);

        float uncertainty2(const Eigen::Matrix<double,2,1> &p2D);

        Eigen::Vector3f unprojectEig(const cv::Point2f &p2D);
        cv::Point3f unproject(const cv::Point2f &p2D);

        Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D);


        bool ReconstructWithTwoViews(const std::shared_ptr<std::vector<KeyPoint>> vKeys1, const std::shared_ptr<std::vector<KeyPoint>> vKeys2, const std::vector<int> &vMatches12,
                                     Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

        cv::Mat toK();
        Eigen::Matrix3f toK_();

        bool epipolarConstrain(GeometricCamera* pCamera2, const KeyPoint& kp1, const KeyPoint& kp2, const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, const float unc);

        float TriangulateMatches(GeometricCamera* pCamera2, const KeyPoint& kp1, const KeyPoint& kp2,  const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float sigmaLevel, const float unc, Eigen::Vector3f& p3D);

        std::vector<int> mvLappingArea;

        bool matchAndtriangulate(const KeyPoint& kp1, const KeyPoint& kp2, GeometricCamera* pOther,
                                 Sophus::SE3f& Tcw1, Sophus::SE3f& Tcw2,
                                 const float sigmaLevel1, const float sigmaLevel2,
                                 Eigen::Vector3f& x3Dtriangulated);

        friend std::ostream& operator<<(std::ostream& os, const KannalaBrandt8& kb);
        friend std::istream& operator>>(std::istream& is, KannalaBrandt8& kb);

        float GetPrecision(){ return precision;}

        bool IsEqual(GeometricCamera* pCam);
    private:
        const float precision;

        //Parameters vector corresponds to
        //[fx, fy, cx, cy, k0, k1, k2, k3]

        TwoViewReconstruction* tvr;

        void Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const Eigen::Matrix<float,3,4> &Tcw1,
                         const Eigen::Matrix<float,3,4> &Tcw2, Eigen::Vector3f &x3D);
    };
}

