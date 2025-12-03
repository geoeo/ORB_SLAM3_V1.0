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


#include <Tracking.h>

#include <ORBmatcher.h>
#include <FrameDrawer.h>
#include <Converter.h>
#include <G2oTypes.h>
#include <Optimizer.h>
#include <CameraModels/Pinhole.h>
#include <CameraModels/KannalaBrandt8.h>
#include <MLPnPsolver.h>
#include <GeometricTools.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <tracy.hpp>

using namespace std;

namespace ORB_SLAM3
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const int sensor, Settings* settings, const TrackerParameters& tracker_settings):
    mState(NO_IMAGES_YET), mSensor(sensor), mCurrentFrame(make_shared<Frame>()),mLastFrame(make_shared<Frame>()), mInitialFrame(make_shared<Frame>()), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), 
    mFrameGridRows(tracker_settings.frameGridRows), mFrameGridCols(tracker_settings.frameGridCols), 
    mMaxLocalKFCount(tracker_settings.maxLocalKFCount), mFeatureThresholdForKF(tracker_settings.featureThresholdForKF),
    mMinFrames(0),mMaxFrames(tracker_settings.maxFrames),
    mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mvpInitFrames(30), mpSystem(pSys), mpViewer(nullptr), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), 
    mnFramesToResetIMU(0), mnLastRelocFrameId(0), time_recently_lost(5.0), mImageTimeout(3.0), mRelocCount(0), mRelocThresh(10),
    mnInitialFrameId(0), mbCreatedMap(false),mLastFramePostDelta(Sophus::SE3f()), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{

    newParameterLoader(settings);
   
    initID = 0; lastID = 0;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    Verbose::PrintMess("There are " + to_string(vpCams.size()) +" cameras in the atlas", Verbose::VERBOSITY_NORMAL);
    for(GeometricCamera* pCam : vpCams)
    {
        Verbose::PrintMess("Camera " + to_string(pCam->GetId()), Verbose::VERBOSITY_NORMAL);
    }

}

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nFastFeatures = settings->nFastFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();
    cv::Size newImSize = settings->newImSize();

    mpORBextractorLeft = new ORBextractor(nFeatures,nFastFeatures ,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, newImSize.width, newImSize.height);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}

tuple<Sophus::SE3f,unsigned long int, bool> Tracking::GrabImageMonocular(const cv::cuda::HostMem &im_managed, const double &timestamp, string filename, bool hasGNSS, Eigen::Vector3f GNSSPosition)
{
    ZoneNamedN(GrabImageMonocular, "GrabImageMonocular", true); 
    assert(mSensor == System::IMU_MONOCULAR);

    mCurrentFrame = std::make_shared<Frame>(im_managed,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,mFrameGridRows, mFrameGridCols, hasGNSS, GNSSPosition, mLastFrame,*mpImuCalib);

    if(mCurrentFrame->mNumKeypoints == 0)
        return {Sophus::SE3f(),0,false};
    if(mpViewer){
        im_managed.createMatHeader().copyTo(mImGrayViewer);
    }

    mCurrentFrame->mNameFile = filename;
    mCurrentFrame->mnDataset = mnNumDataset;

    lastID = mCurrentFrame->mnId;
    Track();

    auto is_keyframe = mpReferenceKF ? mCurrentFrame->mnId == mpReferenceKF->mnFrameId : false;
    return {mCurrentFrame->GetPose(),mCurrentFrame->mnId, false};
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{
    ZoneNamedN(PreIntegrate, "PreIntegrate", true); 
    if(!mCurrentFrame->mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame->setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("No IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame->setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                if(m->t<mCurrentFrame->mpPrevFrame->mTimeStamp)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<=mCurrentFrame->mTimeStamp)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true; //TODO: Is this ever relevant because of break?
            }
        }
        if(bSleep)
           this_thread::sleep_for(chrono::microseconds(500));
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        Verbose::PrintMess("Empty IMU measurements vector", Verbose::VERBOSITY_NORMAL);
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame->mImuBias,mCurrentFrame->mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame->mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame->mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame->mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame->mTimeStamp-mCurrentFrame->mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            Verbose::PrintMess("mpImuPreintegratedFromLastKF does not exist", Verbose::VERBOSITY_NORMAL);
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame->mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame->mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame->setIntegrated();
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame->mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated  && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;
        auto b = mpLastKeyFrame->GetImuBias();
        Verbose::PrintMess("KF Bias ax: " + to_string(b.bax) + "  ay: " + to_string(b.bay) + " az: " + to_string(b.baz), Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("KF Bias wx: " + to_string(b.bwx) + "  wy: " + to_string(b.bwy) + " wz: " + to_string(b.bwz), Verbose::VERBOSITY_DEBUG);
        try{
            Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(b));
            Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(b);
            Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(b);
            mCurrentFrame->SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        } catch(...){
            Verbose::PrintMess("Update branch: IMU Prediction crashed!" , Verbose::VERBOSITY_NORMAL);
            return false;
        }


        mCurrentFrame->mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame->mPredBias = mCurrentFrame->mImuBias;
        return true;
    }
    else
    {
        const Eigen::Vector3f twb1 = mLastFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame->GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame->mpImuPreintegratedFrame->dT;
        auto b = mLastFrame->mImuBias;
        Verbose::PrintMess("Bias ax: " + to_string(b.bax) + "  ay: " + to_string(b.bay) + " az: " + to_string(b.baz), Verbose::VERBOSITY_DEBUG);
        Verbose::PrintMess("Bias wx: " + to_string(b.bwx) + "  wy: " + to_string(b.bwy) + " wz: " + to_string(b.bwz), Verbose::VERBOSITY_DEBUG);
        try {
            Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame->mpImuPreintegratedFrame->GetDeltaRotation(b));
            Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame->mpImuPreintegratedFrame->GetDeltaPosition(b);
            Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame->mpImuPreintegratedFrame->GetDeltaVelocity(b);
            mCurrentFrame->SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        } catch(...){
            Verbose::PrintMess("No Update branch: IMU Prediction crashed!" , Verbose::VERBOSITY_NORMAL);
            return false;
        }

        mCurrentFrame->mImuBias = mLastFrame->mImuBias;
        mCurrentFrame->mPredBias = mCurrentFrame->mImuBias;
        return true;
    }

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}

void Tracking::Track()
{
    ZoneNamedN(Track, "Track", true);
    if (bStepByStep)
    {
        Verbose::PrintMess("Tracking: Waiting to the next step", Verbose::VERBOSITY_NORMAL);
        while(!mbStep && bStepByStep)
           this_thread::sleep_for(chrono::microseconds(500));
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu)
    {
        Verbose::PrintMess("TRACK: Reset map because local mapper set the bad imu flag", Verbose::VERBOSITY_NORMAL);
        mpSystem->Reset();
        return;
    }

    //Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!mpAtlas->GetCurrentMap())
    {
        Verbose::PrintMess("ERROR: There is not an active map in the atlas", Verbose::VERBOSITY_NORMAL);
    }

    if(getTrackingState()!=NO_IMAGES_YET)
    {
        if(mLastFrame->mTimeStamp>mCurrentFrame->mTimeStamp)
        {
            Verbose::PrintMess("ERROR: Frame with a timestamp older than previous frame detected!", Verbose::VERBOSITY_NORMAL);
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame->mTimeStamp>mLastFrame->mTimeStamp+mImageTimeout)
        {
            Verbose::PrintMess("Timestamp image jump detected", Verbose::VERBOSITY_NORMAL);
            setTrackingState(LOST);
        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame->SetNewBias(mpLastKeyFrame->GetImuBias());

    if(getTrackingState()==NO_IMAGES_YET)
    {
        setTrackingState(NOT_INITIALIZED);
    }

    mLastProcessedState=getTrackingState();

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
        PreintegrateIMU();
    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = mpAtlas->GetCurrentMap()->GetMapChangeIndex();
    int nMapChangeIndex = mpAtlas->GetCurrentMap()->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        mpAtlas->GetCurrentMap()->SetLastMapChange(nCurMapChangeIndex);
        Verbose::PrintMess("TRACK: Map has been updated. MapChangeIndex: " + to_string(nCurMapChangeIndex), Verbose::VERBOSITY_DEBUG);
        mbMapUpdated = true;
    }


    if(getTrackingState()==NOT_INITIALIZED)
    {
        Verbose::PrintMess("TRACK: Init ", Verbose::VERBOSITY_DEBUG);
        MonocularInitialization();
        
        //mpFrameDrawer->Update(this);

        if(getTrackingState()!=OK) // If rightly initialized, mState=OK
            Verbose::PrintMess("TRACK: Init was not OK", Verbose::VERBOSITY_DEBUG);
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;


        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(getTrackingState()==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if((!mpAtlas->GetCurrentMap()->isImuInitialized()) || mCurrentFrame->mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                    	bOK = TrackReferenceKeyFrame();
                }


                if (!bOK)
                {

                    setTrackingState(RECENTLY_LOST);
                    mTimeStampLost = mCurrentFrame->mTimeStamp;
                }
            }


            if (getTrackingState() == RECENTLY_LOST)
            {
                Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);
                bOK = false;
                if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                {
                    if(mpAtlas->GetCurrentMap()->isImuInitialized() && (mRelocCount < mRelocThresh)){
                        const auto imu_preint = PredictStateIMU();
                        if(imu_preint){
                            bOK = Relocalization();
                            mRelocCount++;
                            if(bOK){
                                mCurrentFrame->mpPrevFrame = mLastFrame;
                                mRelocCount = 0;
                            }
                        }
                    } else {
                        setTrackingState(LOST);
                    }
                } 
                else {
                    setTrackingState(LOST);
                }

            }
        }

        if(!mCurrentFrame->mpReferenceKF)
            mCurrentFrame->mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMap();
                if(!bOK){
                    Verbose::PrintMess("Fail to track local map!", Verbose::VERBOSITY_NORMAL);
                    setTrackingState(LOST);

                }

            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            setTrackingState(OK);
        else if (getTrackingState() == OK)
        {
            // if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            // {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!mpAtlas->GetCurrentMap()->isImuInitialized())
                {
                    Verbose::PrintMess("IMU is not or recently initialized. Reseting active map..", Verbose::VERBOSITY_NORMAL);
                    setTrackingState(LOST);
                }
                else
                    setTrackingState(RECENTLY_LOST);
            // }
            // else
            //     setTrackingState(RECENTLY_LOST); // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame->mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it should be once mCurrFrame is completely modified)
        // if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
        //    (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpAtlas->GetCurrentMap()->isImuInitialized())
        // {
/*             // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame); */
        // }

        // if(mpAtlas->GetCurrentMap()->isImuInitialized())
        // {
        //     if(bOK)
        //     {
        //         if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
        //         {
        //             Verbose::PrintMess("resetting Imu Frame", Verbose::VERBOSITY_NORMAL);
        //             ResetFrameIMU();
        //         }
        //         else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
        //             mLastBias = mCurrentFrame.mImuBias;
        //     }
        // }

        // Update drawer
        if(mpViewer)
            mpFrameDrawer->Update(this);

        if(mCurrentFrame->isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame->GetPose());

        if(bOK || getTrackingState()==RECENTLY_LOST)
        {
            // Update "motion model"
            if(mLastFrame->isSet() && mCurrentFrame->isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame->GetPose().inverse();
                mLastFramePostDelta = mCurrentFrame->GetPose() * LastTwc;
            }


            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame->GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame->mNumKeypoints; i++)
            {
                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame->mvbOutlier[i] = false;
                        mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(auto pMP: mlpTemporalPoints)
                delete pMP;
                
            mlpTemporalPoints.clear();

            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bNeedKF && (bOK || (mInsertKFsLost && getTrackingState()==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();


            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame->mNumKeypoints;i++)
            {
                if(mCurrentFrame->mvpMapPoints[i] && mCurrentFrame->mvbOutlier[i])
                    mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(getTrackingState()==LOST)
        {
            mpSystem->Reset();
        }

        if(!mCurrentFrame->mpReferenceKF)
            mCurrentFrame->mpReferenceKF = mpReferenceKF;

        //mLastFrame = std::make_shared<Frame>(mCurrentFrame);
    }




    mLastFrame = make_shared<Frame>(mCurrentFrame);
    Verbose::PrintMess("Tracking State:  " + to_string(getTrackingState()), Verbose::VERBOSITY_NORMAL);
}


void Tracking::MonocularInitialization()
{
    ZoneNamedN(MonocularInitialization, "MonocularInitialization", true); 
    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame->mvKeysUn->size()>FEAT_INIT_COUNT)
        {

            mInitialFrame = std::make_shared<Frame>(mCurrentFrame);
            mInitialFrame->SetPose(Sophus::SE3f());

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            Verbose::PrintMess("Ready to initialize", Verbose::VERBOSITY_DEBUG);
            mvpInitFrames.push_back(mInitialFrame);
            mbReadyToInitializate = true;
        }
    }
    else if(mCurrentFrame->mvKeysUn->size()>=FEAT_INIT_COUNT)
    {
        // If time difference is too large, reset init try
        const auto timeDiff = mCurrentFrame->mTimeStamp-mInitialFrame->mTimeStamp;
        mvpInitFrames.push_back(mCurrentFrame);
        if (timeDiff > 2.0f) // 2 seconds
        {
            mbReadyToInitializate = false;
            mvpInitFrames.clear();
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        const auto windowSize = 30; // This parameter has to be large enough to ensure a good disparity but size will impact performance
        auto [nmatches, vIniMatches] = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,windowSize);

        // Check if there are enough correspondences
        if(nmatches<FEAT_INIT_COUNT)
            return;

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (vIniMatches)
        vector<cv::Point3f> vIniP3D; // Initial Correspondences in 3D
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame->mvKeysUn,mCurrentFrame->mvKeysUn,vIniMatches,Tcw,vIniP3D,vbTriangulated))
        {
            Verbose::PrintMess("init matches before 2 view " + to_string(nmatches), Verbose::VERBOSITY_DEBUG);
            
            for(size_t i=0, iend=vIniMatches.size(); i<iend;i++)
            {
                if(vIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    vIniMatches[i]=-1;
                    nmatches--;
                }
            }

            Verbose::PrintMess("init matches after 2 view " + to_string(nmatches), Verbose::VERBOSITY_DEBUG);

            // Set Frame Pose
            mCurrentFrame->SetPose(Tcw);

            CreateInitialMapMonocular(vIniMatches,vIniP3D);
            Verbose::PrintMess("Successful Init: " + to_string(mvpInitFrames.size()), Verbose::VERBOSITY_NORMAL);
        }
    }
}



void Tracking::CreateInitialMapMonocular(const vector<int> &vIniMatches, const vector<cv::Point3f> &vIniP3D)
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    Verbose::PrintMess("CreateInitialMapMonocular init matches: " + to_string(vIniMatches.size()), Verbose::VERBOSITY_DEBUG);

    for(size_t i=0; i<vIniMatches.size();i++)
    {
        if(vIniMatches[i]<0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << vIniP3D[i].x, vIniP3D[i].y, vIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,vIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,vIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateDepth();

        //Fill Current Frame structure
        mCurrentFrame->mvpMapPoints[vIniMatches[i]] = pMP;
        mCurrentFrame->mvbOutlier[vIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_DEBUG);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
        mpSystem->Reset();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame->SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame->mnId;
    mpLastKeyFrame = pKFcur;
    mnLastRelocFrameId = mInitialFrame->mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame->mpReferenceKF = pKFcur;

    mLastFramePostDelta = Sophus::SE3f();

    mLastFrame = std::make_shared<Frame>(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    if(mpViewer)
        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    setTrackingState(OK);

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame->mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame->mnId+1;
    setTrackingState(NO_IMAGES_YET);

    // Restart the variable with information about the last KF
    mLastFramePostDelta = Sophus::SE3f();
    mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mvpInitFrames.clear();
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = std::make_shared<Frame>();
    mCurrentFrame = std::make_shared<Frame>();
    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame->mNumKeypoints; i++)
    {
        MapPoint* pMP = mLastFrame->mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame->mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    ZoneNamedN(TrackReferenceKeyFrame, "TrackReferenceKeyFrame", true); 
    // Compute Bag of Words vector
    mCurrentFrame->ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.95,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<10)
    {
        Verbose::PrintMess("TRACK_REF_KF: Less than 10 matches - " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);
        return false;
    }

    mCurrentFrame->mvpMapPoints = vpMapPointMatches;
    mCurrentFrame->SetPose(mLastFrame->GetPose());

    //mCurrentFrame.PrintPointDistribution();


    Optimizer::PoseOptimization(mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    
    for(int i =0; i<mCurrentFrame->mNumKeypoints; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(mCurrentFrame->mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];

                mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame->mvbOutlier[i]=false;
                if(i < mCurrentFrame->Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame->mnId;
                nmatches--;
            }
            else if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackWithMotionModel()
{
    ZoneNamedN(TrackWithMotionModel, "TrackWithMotionModel", true); 
    ORBmatcher matcher(0.95,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    auto pred_success = false;
    if (mpAtlas->isImuInitialized())
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        Verbose::PrintMess("TrackWithMotionModel - Predict IMU state", Verbose::VERBOSITY_DEBUG);
        pred_success = PredictStateIMU();
    }

    if(pred_success)
         return true;
    else 
        mCurrentFrame->SetPose(mLastFramePostDelta * mLastFrame->GetPose()); // Linear continuation of last frame motion
    


    fill(mCurrentFrame->mvpMapPoints.begin(),mCurrentFrame->mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=10;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("TrackWithMotionModel: Not enough matches, wider window search", Verbose::VERBOSITY_DEBUG);
        fill(mCurrentFrame->mvpMapPoints.begin(),mCurrentFrame->mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("TrackWithMotionModel: Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame->mNumKeypoints; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(mCurrentFrame->mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];

                mCurrentFrame->mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame->mvbOutlier[i]=false;
                if(i < mCurrentFrame->Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame->mnId;
                nmatches--;
            }
            else if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    Verbose::PrintMess("TackWithMotionModel - nmatchesMap: " + to_string(nmatchesMap), Verbose::VERBOSITY_DEBUG);
    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    } else{
        return nmatchesMap>=10;
    }
}

bool Tracking::TrackLocalMap()
{
    ZoneNamedN(TrackLocalMap, "TrackLocalMap", true); 

    UpdateLocalMap();
    SearchLocalPoints();

    // TOO check outliers before PO
    // int aux1 = 0, aux2=0;
    // for(int i=0; i<mCurrentFrame.mNumKeypoints; i++)
    //     if( mCurrentFrame.mvpMapPoints[i])
    //     {
    //         aux1++;
    //         if(mCurrentFrame.mvbOutlier[i])
    //             aux2++;
    //     }

    const auto inlierImuThreshold = 8;
    int inliers;
    if (!mpAtlas->isImuInitialized()){
        inliers = Optimizer::PoseOptimization(mCurrentFrame);
        Verbose::PrintMess("inliers last frame:  " + to_string(inliers), Verbose::VERBOSITY_DEBUG);
    }
    else
    {
        const auto state = getTrackingState();
        if(state==RECENTLY_LOST || state==LOST)
        {
            Verbose::PrintMess("TLM: PoseOptimization - LOST", Verbose::VERBOSITY_DEBUG);
            inliers = Optimizer::PoseOptimization(mCurrentFrame);
            Verbose::PrintMess("inliers last frame:  " + to_string(inliers), Verbose::VERBOSITY_DEBUG);
        }
        else
        {
            const auto prevFrameExists = nullptr != mCurrentFrame->mpPrevFrame;
            const auto mpcpiExists = prevFrameExists ? mCurrentFrame->mpPrevFrame->mpcpi != nullptr : false;

            auto pose_opt_success = false;
            if(!mbMapUpdated && mpcpiExists)
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(mCurrentFrame, inlierImuThreshold);
                Verbose::PrintMess("inliers last frame:  " + to_string(inliers), Verbose::VERBOSITY_DEBUG);
                if(inliers < inlierImuThreshold){
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(mCurrentFrame);
                    Verbose::PrintMess("2# inliers last key frame:  " + to_string(inliers), Verbose::VERBOSITY_NORMAL);
                }
          
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(mCurrentFrame);
                Verbose::PrintMess("inliers last key:  " + to_string(inliers), Verbose::VERBOSITY_DEBUG);
                if(inliers < inlierImuThreshold && mpcpiExists){
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(mCurrentFrame, inlierImuThreshold);  
                    Verbose::PrintMess("2# inliers last frame:  " + to_string(inliers), Verbose::VERBOSITY_DEBUG);
                }

            }
        }
    }

    // aux1 = 0, aux2 = 0;
    // for(int i=0; i<mCurrentFrame.mNumKeypoints; i++)
    //     if( mCurrentFrame.mvpMapPoints[i])
    //     {
    //         aux1++;
    //         if(mCurrentFrame.mvbOutlier[i])
    //             aux2++;
    //     }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame->mNumKeypoints; i++)
    {
        if(mCurrentFrame->mvpMapPoints[i])
        {
            if(!mCurrentFrame->mvbOutlier[i])
            {
                mCurrentFrame->mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame->mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame->mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    //mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    // if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<30)
    //     return false;

    if((mnMatchesInliers>10)&&(getTrackingState()==RECENTLY_LOST))
        return true;

    //if (mSensor == System::IMU_MONOCULAR)
    //{
    const auto pred = (mnMatchesInliers<inlierImuThreshold && mpAtlas->isImuInitialized())||(mnMatchesInliers<30 && !mpAtlas->isImuInitialized());
    return !pred;
    // }
    // else
    // {
    //     return mnMatchesInliers>=30;
    // }
}

bool Tracking::NeedNewKeyFrame()
{
    // if((mSensor == System::IMU_MONOCULAR) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    // {
    //     if ((mSensor == System::IMU_MONOCULAR) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
    //         return true;
    //     else
    //         return false;
    // }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    // {
    //     return false;
    // }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);


    // Thresholds
    float thRefRatio = 0.55;
    if(nKFs<2)
        thRefRatio = 0.4f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1 = mCurrentFrame->mnId>=mnLastKeyFrameId+mMaxFrames;
    const bool c2 = (mnMatchesInliers<nRefMatches*thRefRatio) && (mnMatchesInliers>15);
    const bool c4 = (mnMatchesInliers<mFeatureThresholdForKF) || getTrackingState()==RECENTLY_LOST;

    Verbose::PrintMess("NeedNewKeyFrame: c1 " + to_string(c1) + " c2 " + to_string(c2)+ " c4 " + to_string(c4), Verbose::VERBOSITY_DEBUG);
    return c1 || c4;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame->mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame->mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);

    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame->mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame->mvpMapPoints.begin(), vend=mCurrentFrame->mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame->mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(pMP->mnLastFrameSeen == mCurrentFrame->mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame->isInFrustum(pMP,0.25))
        {
            pMP->IncreaseVisible();
            nToMatch++;
            //TODO: only used in viewer
            mCurrentFrame->mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    Verbose::PrintMess("points to match: " + to_string(nToMatch), Verbose::VERBOSITY_DEBUG);

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.95, true);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            th=5;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // // If the camera has been relocalised recently, perform a coarser search
        // if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
        //     th=10;

        const auto state = getTrackingState();
        if(state==LOST || state==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        auto matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, false, mpLocalMapper->mThFarPoints);
        Verbose::PrintMess("SearchLocalPoints matches: " +to_string(matches), Verbose::VERBOSITY_DEBUG);
    }
}

void Tracking::UpdateLocalMap()
{
    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();

    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame->mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame->mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<unsigned long int,int> keyframeCounter;
    map<unsigned long int,KeyFrame*> keyframePointerMap;

    if(!mpAtlas->isImuInitialized())
    {
        for(int i=0; i<mCurrentFrame->mNumKeypoints; i++)
        {
            MapPoint* pMP = mCurrentFrame->mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++){
                        keyframeCounter[it->first->mnId]++;
                        keyframePointerMap[it->first->mnId]=it->first;
                    }
                }
                else
                {
                    mCurrentFrame->mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame->mNumKeypoints; i++)
        {
            // Using lastframe since current frame has not matches yet due to early exit in TrackWithMotionModel
            if(mLastFrame->mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame->mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++){
                        keyframeCounter[it->first->mnId]++;
                        keyframePointerMap[it->first->mnId]=it->first;
                    }
                }
                else
                {
                    // MODIFICATION
                    mLastFrame->mvpMapPoints[i]=NULL;
                }
            }
        }
    }

    vector <pair<KeyFrame*,int>> vPairs;
    for(map<unsigned long int,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        auto pKFId = it->first;
        auto pKF = keyframePointerMap[pKFId];
        vPairs.push_back(make_pair(pKF,it->second));
    }

    std::sort(vPairs.begin(), vPairs.end(), [](auto& a, auto& b)
    {
        return a.second > b.second;
    });

    const auto nKFs = min(mMaxLocalKFCount,vPairs.size());
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(nKFs);

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(size_t i=0; i<nKFs; i++)
    {
        const auto pKF = vPairs[i].first;

        if(pKF->isBad())
            continue;

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame->mnId;
    }

    Verbose::PrintMess("UpdateLocalKeyFrames: Local KeyFrames: " + to_string(mvpLocalKeyFrames.size()), Verbose::VERBOSITY_DEBUG);

}

bool Tracking::Relocalization()
{
    vector<KeyFrame*> vpCandidateKFs;
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_DEBUG);
    // Compute Bag of Words Vector
    mCurrentFrame->ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(mCurrentFrame, mpAtlas->GetCurrentMap());


    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not enough candidates", Verbose::VERBOSITY_DEBUG);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.85,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            Verbose::PrintMess("Reloc SearchByBoW - Matches:  " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame->SetPose(Tcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame->mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame->mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(mCurrentFrame);
                Verbose::PrintMess("Reloc PoseOptimization - Good:  " + to_string(nGood), Verbose::VERBOSITY_DEBUG);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame->mNumKeypoints; io++)
                    if(mCurrentFrame->mvbOutlier[io])
                        mCurrentFrame->mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);
                    Verbose::PrintMess("Reloc SearchByProjection - Additional:  " + to_string(nadditional), Verbose::VERBOSITY_DEBUG);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(mCurrentFrame);
                        Verbose::PrintMess("Reloc PoseOptimization (2) - Good:  " + to_string(nGood), Verbose::VERBOSITY_DEBUG);


                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame->mNumKeypoints; ip++)
                                if(mCurrentFrame->mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame->mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);
                            Verbose::PrintMess("Reloc SearchByProjection (2)- Additional:  " + to_string(nadditional), Verbose::VERBOSITY_DEBUG);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(mCurrentFrame);
                                Verbose::PrintMess("Reloc PoseOptimization (F)- Good:  " + to_string(nGood), Verbose::VERBOSITY_DEBUG);

                                for(int io =0; io<mCurrentFrame->mNumKeypoints; io++)
                                    if(mCurrentFrame->mvbOutlier[io])
                                        mCurrentFrame->mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        Verbose::PrintMess("Relocalized Failed", Verbose::VERBOSITY_NORMAL);
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame->mnId;
        Verbose::PrintMess("Relocalized Success", Verbose::VERBOSITY_NORMAL);
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
           this_thread::sleep_for(chrono::microseconds(3000));
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    //mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    setTrackingState(NO_IMAGES_YET);

    mvpInitFrames.clear();
    mbReadyToInitializate = false;
    mbSetInit=false;

    mCurrentFrame = std::make_shared<Frame>();
    mnLastRelocFrameId = 0;
    mRelocCount = 0;
    mLastFrame = std::make_shared<Frame>();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("End reseting! ", Verbose::VERBOSITY_NORMAL);
}


// Unsued, since we only have one map. Only difference to Reset() is the mAtlas, mpKeyFrameDB call -> Merge with Reset() function
void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            this_thread::sleep_for(chrono::microseconds(3000));
    }

    
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    //Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    //mpLoopClosing->RequestResetActiveMap(pMap);
    //Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    Map* pMap = mpAtlas->GetCurrentMap();
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    mnLastRelocFrameId = mnLastInitFrameId;
    setTrackingState(NO_IMAGES_YET); //NOT_INITIALIZED;

    mvpInitFrames.clear();
    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = numeric_limits<unsigned int>::max();
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames(false).size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }


    mnInitialFrameId = mCurrentFrame->mnId;
    mnLastRelocFrameId = mCurrentFrame->mnId;

    mCurrentFrame = std::make_shared<Frame>();
    mLastFrame = std::make_shared<Frame>();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mRelocCount = 0;

    mLastFramePostDelta = Sophus::SE3f();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const Sophus::Sim3f &Sim3_Tyw, const IMU::Bias &b)
{
    mLastBias = b;

    mLastFrame->SetNewBias(mLastBias);
    mCurrentFrame->SetNewBias(mLastBias);

    const auto Tyw = Sophus::SE3f(Sim3_Tyw.quaternion(), Sim3_Tyw.translation());
    const auto scale = Sim3_Tyw.scale();

    Verbose::PrintMess("UpdateFrameIMU velocity norm: " + to_string(mLastFrame->mpLastKeyFrame->GetVelocity().norm()), Verbose::VERBOSITY_NORMAL);

    // Important: The scale factor is implcicit in the last keyframe data! Therefore this function has to be called after Map::ApplyScaledRotation!

    if(mLastFrame->imuIsPreintegrated())
    {
        if(mLastFrame->mnId == mLastFrame->mpLastKeyFrame->mnFrameId)
        {
            // dt is 0, therefore we can simplify
            mLastFrame->SetImuPoseVelocity(mLastFrame->mpLastKeyFrame->GetImuRotation(),
                                        mLastFrame->mpLastKeyFrame->GetImuPosition(),
                                        mLastFrame->mpLastKeyFrame->GetVelocity());
        }
        else
        {
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const Eigen::Vector3f twb1 = mLastFrame->mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mLastFrame->mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mLastFrame->mpLastKeyFrame->GetVelocity();
            float t12 = mLastFrame->mpImuPreintegrated->dT;
            mLastFrame->SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame->mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                        twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame->mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                        Vwb1 + Gz*t12 + Rwb1*mLastFrame->mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }
    }
    else {
        Sophus::SE3f Twc = mLastFrame->GetPoseInverse();
        Twc.translation() *= scale;
        Sophus::SE3f Tyc = Tyw*Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        mLastFrame->SetPose(Tcy);
    }


    if(mCurrentFrame->imuIsPreintegrated()){
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mCurrentFrame->mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame->mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame->mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame->mpImuPreintegrated->dT;

        mCurrentFrame->SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame->mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                        twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame->mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                        Vwb1 + Gz*t12 + Rwb1*mCurrentFrame->mpImuPreintegrated->GetUpdatedDeltaVelocity());
    } else {
        Sophus::SE3f Twc = mCurrentFrame->GetPoseInverse();
        Twc.translation() *= scale;
        Sophus::SE3f Tyc = Tyw*Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        mCurrentFrame->SetPose(Tcy);
    }
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

bool Tracking::isBACompleteForMap() {
    return mpAtlas->isBACompleteForMap() || mSensor == System::MONOCULAR;   
}

vector<float> Tracking::getMapScales() {
    return mpAtlas->getMapScales();   
}

void Tracking::setTrackingState(eTrackingState newState){
    unique_lock<mutex> lock(mTrackingState);
    mState = newState;
}

Tracking::eTrackingState Tracking::getTrackingState() {
    unique_lock<mutex> lock(mTrackingState);
    return mState;
}

KeyFrame* Tracking::GetLastKeyFrame() {
    return mpLastKeyFrame;
}

unsigned int Tracking::GetLastKeyFrameId() const {
    return mnLastKeyFrameId;
}

} //namespace ORB_SLAM
