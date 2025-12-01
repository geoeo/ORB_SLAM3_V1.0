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

#include <Frame.h>

#include <G2oTypes.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <ORBextractor.h>
#include <Converter.h>
#include <ORBmatcher.h>
#include <Verbose.h>

#include <thread>
#include <tracy.hpp>
#include <CameraModels/Pinhole.h>
#include <CameraModels/KannalaBrandt8.h>
#include <CameraModels/GeometricCamera.h>

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=1;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(nullptr), mpImuPreintegratedFrame(NULL), 
    mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false),
    mGNSSPosition(Eigen::Vector3f::Zero()), mbHasGNSS(false), mbHasVelocity(false),
    mFrameGridRows(0), mFrameGridCols(0)
{
}


//Copy Constructor
Frame::Frame(const shared_ptr<Frame> frame)
    :mpcpi(frame->mpcpi),mpORBvocabulary(frame->mpORBvocabulary), mpORBextractorLeft(frame->mpORBextractorLeft), mpORBextractorRight(frame->mpORBextractorRight),
     mTimeStamp(frame->mTimeStamp), mK(frame->mK.clone()), mK_(Converter::toMatrix3f(frame->mK)), mDistCoef(frame->mDistCoef.clone()),
     mbf(frame->mbf), mb(frame->mb), mThDepth(frame->mThDepth), mNumKeypoints(frame->mNumKeypoints), mvKeys(frame->mvKeys),
     mvKeysRight(frame->mvKeysRight), mvKeysUn(frame->mvKeysUn), mvuRight(frame->mvuRight),
     mvDepth(frame->mvDepth), mBowVec(frame->mBowVec), mFeatVec(frame->mFeatVec),
     mDescriptors(frame->mDescriptors.clone()), mDescriptorsRight(frame->mDescriptorsRight.clone()),
     mvpMapPoints(frame->mvpMapPoints), mvbOutlier(frame->mvbOutlier), mImuCalib(frame->mImuCalib), mnCloseMPs(frame->mnCloseMPs),
     mpImuPreintegrated(frame->mpImuPreintegrated), mpImuPreintegratedFrame(frame->mpImuPreintegratedFrame), mImuBias(frame->mImuBias),
     mnId(frame->mnId), mpReferenceKF(frame->mpReferenceKF), mnScaleLevels(frame->mnScaleLevels),
     mfScaleFactor(frame->mfScaleFactor), mfLogScaleFactor(frame->mfLogScaleFactor),
     mvScaleFactors(frame->mvScaleFactors), mvInvScaleFactors(frame->mvInvScaleFactors), mNameFile(frame->mNameFile), mnDataset(frame->mnDataset),
     mvLevelSigma2(frame->mvLevelSigma2), mvInvLevelSigma2(frame->mvInvLevelSigma2), mpPrevFrame(frame->mpPrevFrame), mpLastKeyFrame(frame->mpLastKeyFrame),
     mbIsSet(frame->mbIsSet), mbImuPreintegrated(frame->mbImuPreintegrated),
     mFrameGridRows(frame->mFrameGridRows), mFrameGridCols(frame->mFrameGridCols),
     mpCamera(frame->mpCamera), mpCamera2(frame->mpCamera2), Nleft(frame->Nleft), Nright(frame->Nright),
     monoLeft(frame->monoLeft), monoRight(frame->monoRight), mvLeftToRightMatch(frame->mvLeftToRightMatch),
     mvRightToLeftMatch(frame->mvRightToLeftMatch), mvStereo3Dpoints(frame->mvStereo3Dpoints),
     mTlr(frame->mTlr), mRlr(frame->mRlr), mtlr(frame->mtlr), mTrl(frame->mTrl),
     mTcw(frame->mTcw), mbHasPose(frame->mbHasPose), mGNSSPosition(frame->mGNSSPosition), mbHasGNSS(frame->mbHasGNSS), mbHasVelocity(frame->mbHasVelocity)
{
    mGrid.insert(mGrid.end(), frame->mGrid.cbegin(), frame->mGrid.cend());

    if(frame->mbHasPose)
        SetPose(frame->GetPose());

    if(frame->HasVelocity())
        SetVelocity(frame->GetVelocity());
    

    mmProjectPoints = frame->mmProjectPoints;
    mmMatchedInImage = frame->mmMatchedInImage;
}

// Frame& Frame::operator=(const Frame& other){
//     // Guard self assignment
//     if (this == &other)
//         return *this;
    
//     mpcpi = other.mpcpi;
//     mpORBvocabulary = other.mpORBvocabulary;
//     mpORBextractorLeft = other.mpORBextractorLeft;
//     mpORBextractorRight = other.mpORBextractorRight;
//     mTimeStamp = other.mTimeStamp;
//     mK = other.mK.clone();
//     mK_ = Converter::toMatrix3f(other.mK);
//     mDistCoef = other.mDistCoef.clone();
//     mbf = other.mbf;
//     mb = other.mb;
//     mThDepth = other.mThDepth;
//     mNumKeypoints = other.mNumKeypoints;
//     mvKeys = other.mvKeys;
//     mvKeysRight = other.mvKeysRight;
//     mvKeysUn = other.mvKeysUn;
//     mvuRight = other.mvuRight;
//     mvDepth = other.mvDepth;
//     mBowVec = other.mBowVec;
//     mFeatVec = other.mFeatVec;
//     mDescriptors = other.mDescriptors.clone();
//     mDescriptorsRight = other.mDescriptorsRight.clone();
//     mvpMapPoints = other.mvpMapPoints;
//     mvbOutlier = other.mvbOutlier;
//     mImuCalib = other.mImuCalib;
//     mnCloseMPs = other.mnCloseMPs;
//     mpImuPreintegrated = other.mpImuPreintegrated;
//     mpImuPreintegratedFrame = other.mpImuPreintegratedFrame;
//     mImuBias = other.mImuBias;
//     mnId = other.mnId;
//     mpReferenceKF = other.mpReferenceKF;
//     mnScaleLevels = other.mnScaleLevels;
//     mfScaleFactor = other.mfScaleFactor;
//     mfLogScaleFactor = other.mfLogScaleFactor;
//     mvScaleFactors = other.mvScaleFactors;
//     mvInvScaleFactors = other.mvInvScaleFactors;
//     mNameFile = other.mNameFile;
//     mnDataset = other.mnDataset;
//     mvLevelSigma2 = other.mvLevelSigma2;
//     mvInvLevelSigma2 = other.mvInvLevelSigma2;
//     mpPrevFrame = other.mpPrevFrame;
//     mpLastKeyFrame = other.mpLastKeyFrame;
//     mbIsSet = other.mbIsSet;
//     mbImuPreintegrated.store(other.mbImuPreintegrated.load());
//     mFrameGridRows = other.mFrameGridRows;
//     mFrameGridCols = other.mFrameGridCols;
//     mpCamera = other.mpCamera;
//     mpCamera2 = other.mpCamera2;
//     Nleft = other.Nleft;
//     Nright = other.Nright;
//     monoLeft = other.monoLeft;
//     monoRight = other.monoRight;
//     mvLeftToRightMatch = other.mvLeftToRightMatch;
//     mvRightToLeftMatch = other.mvRightToLeftMatch;
//     mvStereo3Dpoints = other.mvStereo3Dpoints;
//     mTlr = other.mTlr;
//     mRlr = other.mRlr;
//     mtlr = other.mtlr;
//     mTrl = other.mTrl;
//     mTcw = other.mTcw;
//     mbHasPose = other.mbHasPose;
//     mGNSSPosition = other.mGNSSPosition;
//     mbHasGNSS = other.mbHasGNSS;
//     mbHasVelocity = other.mbHasVelocity;

//     mGrid.insert(mGrid.end(), other.mGrid.begin(), other.mGrid.end());
//     if(other.mbHasPose)
//         SetPose(other.GetPose());

//     if(other.HasVelocity())
//         SetVelocity(other.GetVelocity());
    

//     mmProjectPoints = other.mmProjectPoints;
//     mmMatchedInImage = other.mmMatchedInImage;


//     return *this;
// }


Frame::Frame(const cv::cuda::HostMem &im_managed_gray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, 
    GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth,  int frameGridRows, int frameGridCols,
    bool hasGNSS, Eigen::Vector3f GNSSPosition, std::shared_ptr<Frame> pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), 
     mThDepth(thDepth),mNumKeypoints(0),mFrameGridRows(frameGridRows), mFrameGridCols(frameGridCols), mImuCalib(ImuCalib), 
     mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(nullptr), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mGNSSPosition(GNSSPosition), mbHasGNSS(hasGNSS), mbHasVelocity(false)
{
    ZoneNamedN(Frame, "Frame", true); 
    const auto size = frameGridCols*frameGridRows;
    mGrid.resize(size);

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,im_managed_gray);
    if(mNumKeypoints == 0)
        return;

    mvKeysUn = mvKeys;

    // Set no stereo information
    mvuRight = vector<float>(mNumKeypoints,-1);
    mvDepth = vector<float>(mNumKeypoints,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(mNumKeypoints,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(mNumKeypoints,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(im_managed_gray);

        mfGridElementWidthInv=static_cast<float>(mFrameGridCols)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(mFrameGridRows)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }
}

int Frame::computeLinearGridIndex(int col, int row, int cols) {
    return row*cols + col;    
}

int Frame::getFrameGridRows() const {
    return mFrameGridRows;
}

int Frame::getFrameGridCols() const{
    return mFrameGridCols;
}

void Frame::AssignFeaturesToGrid()
{
    ZoneNamedN(AssignFeaturesToGrid, "AssignFeaturesToGrid", true); 
    // Fill matrix with points
    const int nCells = mFrameGridCols*mFrameGridRows;

    for(int i = 0; i < nCells; ++i)
        mGrid[i].reserve(nCells);


    for(int i=0;i<mNumKeypoints;i++)
    {
        const auto &kp = mvKeysUn->operator[](i);
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            auto linear_index = computeLinearGridIndex(nGridPosX,nGridPosY,mFrameGridCols);
            mGrid[linear_index].push_back(i);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::cuda::HostMem &im_managed)
{
    auto key_desc_optional = mpORBextractorLeft->extractFeatures(im_managed);

    if(key_desc_optional.has_value()){
        auto [keys,descriptors] = key_desc_optional.value();
        monoLeft = keys->size();
        mNumKeypoints = keys->size();
        mvKeys = keys;
        mDescriptors = descriptors;
    }


}

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;
    mTwc = mTcw.inverse();

    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{   
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Verbose::PrintMess("Rwb - " + to_string(Rwb(0,0)) + ", " + to_string(Rwb(1,1)) + ", " + to_string(Rwb(2,2)), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("twb - " + to_string(twb(0)) + ", " + to_string(twb(1)) + ", " + to_string(twb(2)), Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Vwb - " + to_string(Vwb(0)) + ", " + to_string(Vwb(1)) + ", " + to_string(Vwb(2)), Verbose::VERBOSITY_DEBUG);

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;
    mTwc = mTcw.inverse();

    mbIsSet = true;
    mbHasPose = true;
}


Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return GetRwc() * mImuCalib.mTcb.translation() + GetTwc();
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return GetRwc() * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;
    pMP->mTrackProjX = -1;
    pMP->mTrackProjY = -1;

    // 3D in absolute coordinates
    Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Matrix<float,3,1> Pc = GetRcw() * P + GetTcw();
    const float Pc_dist = Pc.norm();

    // Check positive depth
    const float &PcZ = Pc(2);
    const float invz = 1.0f/PcZ;
    if(PcZ<0.0f)
        return false;

    const Eigen::Vector2f uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    pMP->mTrackProjX = uv(0);
    pMP->mTrackProjY = uv(1);

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - GetTwc();
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = uv(0);
    pMP->mTrackProjXR = uv(0) - mbf*invz;
    pMP->mTrackDepth = Pc_dist;

    pMP->mTrackProjY = uv(1);
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = GetRcw() * P + GetTcw();
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return GetRcw() * pCw + GetTcw();
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    ZoneNamedN(GetFeaturesInArea, "GetFeaturesInArea", true); 
    vector<size_t> vIndices;
    vIndices.reserve(mNumKeypoints);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=mFrameGridCols)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)mFrameGridCols-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=mFrameGridRows)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)mFrameGridRows-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            auto linear_index = computeLinearGridIndex(ix,iy,mFrameGridCols);
            const auto vCell = mGrid[linear_index];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const auto &kpUn = mvKeysUn->operator[](vCell[j]);
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel || (maxLevel>=0 && kpUn.octave>maxLevel)){
                        continue;
                    }
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    auto linearIdx = computeLinearGridIndex(posX,posY,mFrameGridCols);
    const auto size = mFrameGridCols*mFrameGridRows;

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    return linearIdx >= 0 & linearIdx<size;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        ZoneNamedN(ComputeBoW, "ComputeBoW", true);
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors.createMatHeader());
        {
            ZoneNamedN(ComputeBoWTransform, "ComputeBoWTransform", true);
            mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        }
        
    }
}

void Frame::ComputeImageBounds(const cv::cuda::HostMem &imLeftManaged)
{

    cv::Mat imLeft = imLeftManaged.createMatHeader();

    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

bool Frame::imuIsPreintegrated()
{
    //unique_lock<std::mutex> lock(mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    //unique_lock<std::mutex> lock(mpMutexImu);
    mbImuPreintegrated = true;
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * GetRcw();
        mt = Rrl * GetTcw() + trl;
        twc = GetRwc() * mTlr.translation() + GetTwc();
    }
    else{
        mR = GetRcw();
        mt = GetTcw();
        twc = GetTwc();
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return GetRwc() * mvStereo3Dpoints[i] + GetTwc();
}

} //namespace ORB_SLAM
