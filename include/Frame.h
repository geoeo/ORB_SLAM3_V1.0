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
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/geometry.hpp>

#include <KeyPoint.h>
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <ImuTypes.h>
#include <ORBVocabulary.h>
#include <Converter.h>
#include <Settings.h>

namespace ORB_SLAM3
{

//TODO: Investigate these forward declarations
class MapPoint;
class KeyFrame;
class ConstraintPoseImu;
class GeometricCamera;
class ORBextractor;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for Monocular cameras.
    Frame(const cv::cuda::HostMem &im_managed_gray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, 
        GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, int frameGridRows, int frameGridCols,
        Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Destructor
    // ~Frame();

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::cuda::HostMem &im_managed);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(const Sophus::SE3<float> &Tcw);

    // Set IMU velocity
    void SetVelocity(Eigen::Vector3f Vw);

    Eigen::Vector3f GetVelocity() const;

    // Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);

    Eigen::Matrix<float,3,1> GetImuPosition() const;
    Eigen::Matrix<float,3,3> GetImuRotation();
    Sophus::SE3<float> GetImuPose();

    Sophus::SE3f GetRelativePoseTrl();
    Sophus::SE3f GetRelativePoseTlr();
    Eigen::Matrix3f GetRelativePoseTlr_rotation();
    Eigen::Vector3f GetRelativePoseTlr_translation();

    void SetNewBias(const IMU::Bias &b);

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    Eigen::Vector3f inRefCoordinates(Eigen::Vector3f pCw);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const KeyPoint &kp, int &posX, int &posY);

    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1, const bool bRight = false) const;

    ConstraintPoseImu* mpcpi;

    bool imuIsPreintegrated();
    void setIntegrated();

    bool isSet() const;

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline Eigen::Vector3f GetCameraCenter(){
        return mOw;
    }

    // Returns inverse of rotation
    inline Eigen::Matrix3f GetRotationInverse(){
        return mRwc;
    }

    inline Sophus::SE3<float> GetPose() const {
        //TODO: can the Frame pose be accsessed from several threads? should this be protected somehow?
        return mTcw;
    }

    inline Eigen::Matrix3f GetRwc() const {
        return mRwc;
    }

    inline Eigen::Vector3f GetOw() const {
        return mOw;
    }

    inline bool HasPose() const {
        return mbHasPose;
    }

    inline bool HasVelocity() const {
        return mbHasVelocity;
    }



private:
    //Sophus/Eigen migration
    Sophus::SE3<float> mTcw;
    Eigen::Matrix<float,3,3> mRwc;
    Eigen::Matrix<float,3,1> mOw;
    Eigen::Matrix<float,3,3> mRcw;
    Eigen::Matrix<float,3,1> mtcw;
    bool mbHasPose;

    //Rcw_ not necessary as Sophus has a method for extracting the rotation matrix: Tcw_.rotationMatrix()
    //tcw_ not necessary as Sophus has a method for extracting the translation vector: Tcw_.translation()
    //Twc_ not necessary as Sophus has a method for easily computing the inverse pose: Tcw_.inverse()

    Sophus::SE3<float> mTlr, mTrl;
    Eigen::Matrix<float,3,3> mRlr;
    Eigen::Vector3f mtlr;


    // IMU linear velocity
    Eigen::Vector3f mVw;
    bool mbHasVelocity;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int mNumKeypoints;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::shared_ptr<std::vector<KeyPoint>> mvKeys, mvKeysRight;
    std::shared_ptr<std::vector<KeyPoint>> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    std::vector<MapPoint*> mvpMapPoints;
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::cuda::HostMem mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;
    int mnCloseMPs;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::vector<std::size_t>> mGrid; // Represent a 2D grid [rows][cols] of vectors with 1D index

    IMU::Bias mPredBias;

    // IMU bias
    IMU::Bias mImuBias;

    // Imu calibration
    IMU::Calib mImuCalib;

    // Imu preintegration from last keyframe
    IMU::Preintegrated* mpImuPreintegrated;
    KeyFrame* mpLastKeyFrame;

    // Pointer to previous frame
    Frame* mpPrevFrame;
    IMU::Preintegrated* mpImuPreintegratedFrame;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    std::map<long unsigned int, cv::Point2f> mmProjectPoints;
    std::map<long unsigned int, cv::Point2f> mmMatchedInImage;

    std::string mNameFile;

    int mnDataset;

#ifdef REGISTER_TIMES
    double mTimeORB_Ext;
    double mTimeStereoMatch;
#endif

private:
    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::cuda::HostMem &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    bool mbIsSet;

    bool mbImuPreintegrated;

    std::shared_ptr<std::mutex> mpMutexImu;

    int mFrameGridRows; 
    int mFrameGridCols; 

public:
    GeometricCamera* mpCamera, *mpCamera2;

    //Number of KeyPoints extracted in the left and right images
    int Nleft, Nright;
    //Number of Non Lapping Keypoints
    int monoLeft, monoRight;

    //For stereo matching
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    //For stereo fisheye matching
    static cv::BFMatcher BFmatcher;

    //Triangulated stereo observations using as reference the left camera. These are
    //computed during ComputeStereoFishEyeMatches
    std::vector<Eigen::Vector3f> mvStereo3Dpoints;


    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    static int computeLinearGridIndex(int col, int row, int cols);

    int getFrameGridRows() const;
    
    int getFrameGridCols() const;

    bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);

    Eigen::Vector3f UnprojectStereoFishEye(const int &i);

    cv::Mat imgLeft, imgRight;

    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (Nleft != -1) ? Nleft : mNumKeypoints;
        for(int i = 0; i < mNumKeypoints; i++){
            if(mvpMapPoints[i] && !mvbOutlier[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        std::cout << "Point distribution in Frame: left-> " << left << " --- right-> " << right << std::endl;
    }

    Sophus::SE3<double> T_test;
};

}// namespace ORB_SLAM
