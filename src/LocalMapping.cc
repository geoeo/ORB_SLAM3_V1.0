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


#include <LocalMapping.h>
#include <ORBmatcher.h>
#include <Optimizer.h>
#include <Converter.h>
#include <GeometricTools.h>
#include <Verbose.h>
#include <tracy.hpp>

#include <chrono>
#include <thread>
#include <algorithm>
#include <execution>

using namespace std;

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(std::shared_ptr<Atlas> pAtlas, const float bMonocular, bool bInertial, const LocalMapperParameters &local_mapper):
    mScale(1.0), mInitSect(0),mnMatchesInliers(0), mIdxIteration(0), mbNotBA1(true), mbNotBA2(true), mbBadImu(false),mThFarPoints(local_mapper.thFarPoints), mbFarPoints(mThFarPoints!=0.0f), mbMonocular(bMonocular), 
    mbFixScale(false), mbInertial(bInertial), mbResetRequested(false), 
    mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), mbAbortBA(false), mbStopped(false), mbStopRequested(false), 
    mbNotStop(false), mMutexPtrGlobalData(make_shared<mutex>()), 
    bInitializing(false), infoInertial(Eigen::MatrixXd::Zero(9,9)), mNumLM(0),mNumKFCulling(0), mTElapsedTime(0.0),
    resetTimeThresh(local_mapper.resetTimeThresh), minTimeForImuInit(local_mapper.minTimeForImuInit), 
    minTimeForVIBA1(local_mapper.minTimeForVIBA1), minTimeForVIBA2(local_mapper.minTimeForVIBA2), minTimeForFullBA(local_mapper.minTimeForFullBA),
    itsFIBAInit(local_mapper.itsFIBAInit), itsFIBA1(local_mapper.itsFIBA1),writeKFAfterGeorefCount(0), writeKFAfterGBACount(0),
    mbUseGNSS(local_mapper.useGNSS), mbUseGNSSBA(local_mapper.useGNSSBA), mbWriteGNSSData(local_mapper.writeGNSSData), mbGeorefUpdate(local_mapper.georefUpdate),
    mGeometricReferencer(local_mapper.minGeorefFrames), mLatestOptimizedKFPoses({})
{
}

void LocalMapping::SetLoopCloser(shared_ptr<LoopClosing> pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(shared_ptr<Tracking> pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    mbFinished = false;

    while(!CheckFinish())
    {
        ZoneNamedN(LocalMapping, "LocalMapping", true);  // NOLINT: Profiler

        ResetIfRequested();

        // If we dont use the IMU the initial map space is the final one
        if(!mbInertial){
            mpAtlas->GetCurrentMap()->SetInertialBA1();
            mpAtlas->GetCurrentMap()->SetInertialBA2();
            mpAtlas->GetCurrentMap()->SetInertialFullBA();
        }

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames() && !mbBadImu)
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();
            // Check recent MapPoints
            MapPointCulling();
            // Triangulate new MapPoints
            CreateNewMapPoints();
            
            if(mpAtlas->GetCurrentMap()->GetInertialFullBA() && mbUseGNSS){
                //TODO: This lock leads to deadlock(?) investigate why this happens
                unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
                const auto georef_succcess = GeoreferenceKeyframes();
                if(georef_succcess && writeKFAfterGeorefCount == 0){
                    if(mbWriteGNSSData){
                        const auto kfs = mpAtlas->GetCurrentMap()->GetAllKeyFrames(false);
                        for(const auto kf : kfs)
                            kf->ComputeReprojectionErrors(true);
                        Map::writeKeyframesCsv("keyframes_after_georef", kfs);
                        Map::writeKeyframesReprojectionErrors("reprojections_after_georef", kfs);
                    }

                    writeKFAfterGeorefCount = 1;
                }
                if(mGeometricReferencer.isInitialized() && mbUseGNSSBA && !mbResetRequested){
                    unique_lock<mutex> lockGlobal(*getGlobalDataMutex());
                    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
                    if(writeKFAfterGBACount == 0){
                        Verbose::PrintMess("Starting GNSS Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
                        Optimizer::LocalGNSSBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpAtlas->GetCurrentMap(), mGeometricReferencer);
                        if(mbWriteGNSSData){
                            const auto kfs = mpAtlas->GetCurrentMap()->GetAllKeyFrames(false);
                            Map::writeKeyframesCsv("keyframes_after_gnss_bundle", kfs);
                            Map::writeKeyframesReprojectionErrors("reprojections_after_gnss_bundle", kfs);
                        }

                        //TODO: This is not the correct place for this, as the GeroefUpdate would apply the transformation twice
                        //TODO: Check for submitted but to processed KFs
                        const auto sortedKFs = mpAtlas->GetCurrentMap()->GetAllKeyFrames(true);    
                        const auto georef_transform = mGeometricReferencer.getCurrentTransform().cast<float>();
                        //const Eigen::Quaternionf rotation(Eigen::AngleAxisf(0.5*EIGEN_PI,Eigen::Vector3f::UnitZ()));
                        //const Sophus::Sim3f Sim3_Tyw_noscale(1.0, Sophus::SO3f().unit_quaternion(), georef_transform.translation());
                        //const Sophus::Sim3f Sim3_Tyw_noscale(1.5f, Sophus::SO3f(rotation).unit_quaternion(), Sophus::Vector3f::Zero());
                        //const Sophus::Sim3f Sim3_Tyw_noscale(georef_transform.scale(), georef_transform.rxso3().quaternion().normalized(), Sophus::Vector3f::Zero());
                        const Sophus::Sim3f Sim3_Tyw_noscale(1.0, georef_transform.rxso3().quaternion().normalized(), Sophus::Vector3f::Zero());
                        //TODO: seems to be a problem with translation
                        //const auto b_transformed = georef_transform.rxso3().quaternion()*sortedKFs.front()->GetImuBias();

                        
                        // while(CheckNewKeyFrames())
                        //     ProcessNewKeyFrame();
                        // UpdateTrackerAndMapCoordianateFrames(sortedKFs, Sim3_Tyw_noscale, {});
                        //Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 20, false, 0, NULL, false, 0, 0);



                        // const auto georef_pose = getGeorefTransform();
                        // // Coordiante Frames should be aligned, we only need to set the translation
                        // const auto sortedKfs = mpAtlas->GetCurrentMap()->GetAllKeyFrames(true);
                        // const auto currentFramePointer = mpTracker->GetCurrentFrame();
                        // const auto initialFramePointer = mpTracker->GetInitialFrame();
                        // const auto lastFramePointer = mpTracker->GetLastFrame();
                        // const auto lastKFPointer = mpTracker->GetLastKeyFrame();
                        // const auto firstKf = sortedKfs.front();
                        // const auto lastKF = sortedKfs.back();


                        // const auto deltaPose = initialFramePointer->GetPose()*lastKF->GetPoseInverse();
                        // //const auto gnssDeltaGNSSPose = initialFramePointer->GetGNSSCameraPose().inverse()*lastKF->GetGNSSCameraPose();
                        // const auto poseWorld = lastKF->GetPoseInverse();


                        // const Eigen::Vector3f gnssDeltaTranslation = mpTracker->GetCurrentFrame()->GetGNSS() - mpTracker->GetInitialFrame()->GetGNSS();


                        // Verbose::PrintMess("Transformation matrix:", Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess(to_string(georef_pose.rotationMatrix()(0,0)) + " " + to_string(georef_pose.rotationMatrix()(0,1)) + " " + to_string(georef_pose.rotationMatrix()(0,2)) + " " + to_string(georef_pose.translation()(0)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess(to_string(georef_pose.rotationMatrix()(1,0)) + " " + to_string(georef_pose.rotationMatrix()(1,1)) + " " + to_string(georef_pose.rotationMatrix()(1,2)) + " " + to_string(georef_pose.translation()(1)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess(to_string(georef_pose.rotationMatrix()(2,0)) + " " + to_string(georef_pose.rotationMatrix()(2,1)) + " " + to_string(georef_pose.rotationMatrix()(2,2)) + " " + to_string(georef_pose.translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Scale: " + to_string(georef_pose.scale()), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("\n", Verbose::VERBOSITY_NORMAL);


                        // Verbose::PrintMess("TRACK: Current Poses\n ", Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("X: " +  to_string(currentFramePointer->GetPoseInverse().translation()(0)) + " Y: " + to_string(currentFramePointer->GetPoseInverse().translation()(1)) + " Z: " + to_string(currentFramePointer->GetPoseInverse().translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Latest KF X: " +  to_string(lastKFPointer->GetPoseInverse().translation()(0)) + " Y: " + to_string(lastKFPointer->GetPoseInverse().translation()(1)) + " Z: " + to_string(lastKFPointer->GetPoseInverse().translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Initial Frame X: " +  to_string(initialFramePointer->GetPoseInverse().translation()(0)) + " Y: " + to_string(initialFramePointer->GetPoseInverse().translation()(1)) + " Z: " + to_string(initialFramePointer->GetPoseInverse().translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Initial Frame Quat X: " +  to_string(initialFramePointer->GetPoseInverse().unit_quaternion().x()) + " Y: " + to_string(initialFramePointer->GetPoseInverse().unit_quaternion().y()) + " Z: " + to_string(initialFramePointer->GetPoseInverse().unit_quaternion().z()) + " W: " + to_string(initialFramePointer->GetPoseInverse().unit_quaternion().w()), Verbose::VERBOSITY_NORMAL);

                        // Verbose::PrintMess("Delta KF: " +  to_string(deltaPose.translation()(0)) + " Y: " + to_string(deltaPose.translation()(1)) + " Z: " + to_string(deltaPose.translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // //Verbose::PrintMess("Delta GNSS KF: " +  to_string(gnssDeltaGNSSPose.translation()(0)) + " Y: " + to_string(gnssDeltaGNSSPose.translation()(1)) + " Z: " + to_string(gnssDeltaGNSSPose.translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Pose KF: " +  to_string(poseWorld.translation()(0)) + " Y: " + to_string(poseWorld.translation()(1)) + " Z: " + to_string(poseWorld.translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("\n", Verbose::VERBOSITY_NORMAL);
                        
                        // Verbose::PrintMess("Delta 1 X: " +  to_string(gnssDeltaTranslation(0)) + " Y: " + to_string(gnssDeltaTranslation(1)) + " Z: " + to_string(gnssDeltaTranslation(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Current GNSS X: " +  to_string(currentFramePointer->GetGNSS()(0)) + " Y: " + to_string(currentFramePointer->GetGNSS()(1)) + " Z: " + to_string(currentFramePointer->GetGNSS()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Initial GNSS X: " +  to_string(initialFramePointer->GetGNSS()(0)) + " Y: " + to_string(initialFramePointer->GetGNSS()(1)) + " Z: " + to_string(initialFramePointer->GetGNSS()(2)), Verbose::VERBOSITY_NORMAL);

                        // Verbose::PrintMess("Latest KF GNSS Cam X: " +  to_string(lastKFPointer->GetGNSSCameraPose().translation()(0)) + " Y: " + to_string(lastKFPointer->GetGNSSCameraPose().translation()(1)) + " Z: " + to_string(lastKFPointer->GetGNSSCameraPose().translation()(2)), Verbose::VERBOSITY_NORMAL);
                        // Verbose::PrintMess("Latest KF GNSS X: " +  to_string(lastKFPointer->GetRawGNSSPosition()(0)) + " Y: " + to_string(lastKFPointer->GetRawGNSSPosition()(1)) + " Z: " + to_string(lastKFPointer->GetRawGNSSPosition()(2)), Verbose::VERBOSITY_NORMAL);

                        writeKFAfterGBACount = 1;
                        //throw std::runtime_error("GNSS fallback not implemented yet");
                    }
                }



            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;


            if(mpAtlas->KeyFramesInMap()>2)
            {

                if(mbInertial && mpAtlas->GetCurrentMap()->isImuInitialized())
                {
                    if(!mpAtlas->GetCurrentMap()->GetInertialBA2() && (mTElapsedTime>resetTimeThresh))
                    {
                        Verbose::PrintMess("Not enough motion for initializing. Reseting...", Verbose::VERBOSITY_DEBUG);
                        mbBadImu = true;
                    }

                    Verbose::PrintMess("LocalMapper - LocalInertialBA", Verbose::VERBOSITY_DEBUG);
                    {
                        //unique_lock<mutex> lockGlobal(*getGlobalDataMutex());
                        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
                        auto optimizedKFPoses = Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpAtlas->GetCurrentMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, !mpCurrentKeyFrame->GetMap()->GetInertialBA2());
                        setLatestOptimizedKFPoses(optimizedKFPoses);
                        Verbose::PrintMess("LocalMapper - LocalInertialBA - Abort: " + to_string(mbAbortBA), Verbose::VERBOSITY_DEBUG);
                    }

                    b_doneLBA = true;
                }
                else
                {
                    Verbose::PrintMess("LocalMapper - LocalBundleAdjustment", Verbose::VERBOSITY_DEBUG);
                    {
                        //unique_lock<mutex> lockGlobal(*getGlobalDataMutex());
                        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpAtlas->GetCurrentMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        Verbose::PrintMess("LocalMapper - LocalBundleAdjustment - Abort: " + to_string(mbAbortBA), Verbose::VERBOSITY_DEBUG);
                    }

                    b_doneLBA = true;
                }

            }

            // Initialize IMU here
            if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
            {
                Verbose::PrintMess("Initial IMU Init", Verbose::VERBOSITY_DEBUG);
                auto success = InitializeIMU(1e10, 1e10, true, itsFIBAInit, minTimeForImuInit, 10);
                Verbose::PrintMess("Initial IMU Init Success: " + to_string(success), Verbose::VERBOSITY_DEBUG);
                if(false){
                    mpAtlas->GetCurrentMap()->SetInertialBA1();
                    mpAtlas->GetCurrentMap()->SetInertialBA2();
                    if(minTimeForFullBA < 0)
                        mpAtlas->GetCurrentMap()->SetInertialFullBA();
                }
            }


            // Check redundant local Keyframes -  Disabled for now since it may causes Segfaults: TODO: Fix
            //KeyFrameCulling();

            if (mbInertial)
            {
                if(mpAtlas->GetCurrentMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                {
                    Verbose::PrintMess("check VIBA", Verbose::VERBOSITY_DEBUG);
                    if(!mpAtlas->GetCurrentMap()->GetInertialBA1()){
                        Verbose::PrintMess("start VIBA 1", Verbose::VERBOSITY_DEBUG);
                        auto success = false;
                        if (mbMonocular)
                            success = InitializeIMU(1e10, 1e10, true, itsFIBA1, minTimeForVIBA1, 10);
                        else
                            success = InitializeIMU(1.f, 1e5, true, itsFIBA1, minTimeForVIBA1, 10);

                        if(success){
                            mpAtlas->GetCurrentMap()->SetInertialBA1();
                            mpAtlas->GetCurrentMap()->SetInertialBA2(); // skip second BA
                            if(minTimeForFullBA < 0)
                                mpAtlas->GetCurrentMap()->SetInertialFullBA();
                        }
                        
                        Verbose::PrintMess("end VIBA 1 " + to_string(success), Verbose::VERBOSITY_DEBUG);
                    }
                    if(!mpAtlas->GetCurrentMap()->GetInertialBA2() && mpAtlas->GetCurrentMap()->GetInertialBA1()){
                        Verbose::PrintMess("start VIBA 2", Verbose::VERBOSITY_DEBUG);
                        auto success = false;
                        if (mbMonocular)
                            success = InitializeIMU(0.f, 0.f, false, 200, minTimeForVIBA2, 15); // TODO: priorA is small causes reloc issues. Investigate
                        else
                            success = InitializeIMU(0.f, 0.f, true, 200, minTimeForVIBA2, 10);

                        if(success){
                            mpAtlas->GetCurrentMap()->SetInertialBA2();
                            if(minTimeForFullBA < 0)
                                mpAtlas->GetCurrentMap()->SetInertialFullBA();
                        }
                        Verbose::PrintMess("end VIBA 2 " + to_string(success), Verbose::VERBOSITY_DEBUG);
                    }

                    if(!mbResetRequested && !mpAtlas->GetCurrentMap()->GetInertialBA2() && mTElapsedTime>minTimeForFullBA && minTimeForFullBA >= 0 && !mpAtlas->GetCurrentMap()->GetInertialFullBA()){
                        unique_lock<mutex> lock(*getGlobalDataMutex());
                        Verbose::PrintMess("Full BA Start", Verbose::VERBOSITY_DEBUG);
                        //Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, false, 0.0, 1e2);
                        auto success = InitializeIMU(0.f, 1e2, true, 200,3, 15);
                        if(success)
                            mpAtlas->GetCurrentMap()->SetInertialFullBA();
                    }
                }
            }
            

            //mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            Verbose::PrintMess("LocalMapper - Elapsed Time: " + to_string(mTElapsedTime) + " KF size:" + to_string(mpAtlas->GetCurrentMap()->GetAllKeyFrames(false).size()), Verbose::VERBOSITY_NORMAL);

        }

        if(CheckFinish())
            break;
    }

}

void LocalMapping::InsertKeyFrame(shared_ptr<KeyFrame> pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    //mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ResetNewKeyFrames() 
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.clear();
}

int LocalMapping::KeyframesInQueue()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return mlNewKeyFrames.size();
}

void LocalMapping::ProcessNewKeyFrame()
{
    ZoneNamedN(LocalMapping_ProcessNewKeyFrame, "LocalMapping_ProcessNewKeyFrame", true);  // NOLINT: Profiler
    {
        Verbose::PrintMess("LocalMapper - New KF Sizes: " + to_string(mlNewKeyFrames.size()), Verbose::VERBOSITY_DEBUG);
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        if(mpCurrentKeyFrame->mPrevKF)
            mTElapsedTime += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const auto vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        auto pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    {
        unique_lock<mutex> lock(*getGlobalDataMutex());
        mpCurrentKeyFrame->UpdateConnections();
    }


    if(mpAtlas->isImuInitialized())
        mpCurrentKeyFrame->GetKeyFrameDatabase()->add(mpCurrentKeyFrame);

    // Insert Keyframe in Map
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
    mGeometricReferencer.addKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        auto pMP = *lit;

        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
    }
}


void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    size_t nn = 15;
    // For stereo inertial case
    // if(mbMonocular)
    //     nn=30;
    vector<shared_ptr<KeyFrame>> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (mbInertial)
    {
        shared_ptr<KeyFrame> pKF = mpCurrentKeyFrame;
        size_t count=0;
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {
            vector<shared_ptr<KeyFrame>>::iterator it = find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f twc1 = mpCurrentKeyFrame->GetTranslationInverse();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        auto pKF2 = vpNeighKFs[i];

        auto pCamera1 = mpCurrentKeyFrame->mpCamera;
        auto pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f twc2 = pKF2->GetTranslationInverse();
        Eigen::Vector3f vBaseline = twc2-twc1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01){
                Verbose::PrintMess("LocalMapping::CreateNewMapPoints: Baseline too short: " + to_string(ratioBaselineDepth), Verbose::VERBOSITY_DEBUG);
                continue;
            }
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetInertialBA2();

        // threshold does not seem to affect triangulation
        float th = 0.6f;
        ORBmatcher matcher(th,false);
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const auto &kp1 = mpCurrentKeyFrame->mvKeysUn->operator[](idx1);
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            const bool bRight1 = false;
            const auto &kp2 = pKF2->mvKeysUn->operator[](idx2);

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    twc1 = mpCurrentKeyFrame->GetRightTranslationInverse();

                    sophTcw2 = pKF2->GetRightPose();
                    twc2 = pKF2->GetRightTranslationInverse();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    twc1 = mpCurrentKeyFrame->GetRightTranslationInverse();

                    sophTcw2 = pKF2->GetPose();
                    twc2 = pKF2->GetTranslationInverse();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    twc1 = mpCurrentKeyFrame->GetTranslationInverse();

                    sophTcw2 = pKF2->GetRightPose();
                    twc2 = pKF2->GetRightTranslationInverse();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    twc1 = mpCurrentKeyFrame->GetTranslationInverse();

                    sophTcw2 = pKF2->GetPose();
                    twc2 = pKF2->GetTranslationInverse();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<0.9996 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - twc1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - twc2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            auto pMP = make_shared<MapPoint>(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            if (bPointStereo)
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }    
}

bool LocalMapping::GeoreferenceKeyframes(){
    const auto wasInitialized = mGeometricReferencer.isInitialized();
    // First check if we have any frames which we need to georeference
    const auto kfsWithoutGeoref = wasInitialized ? mGeometricReferencer.getFramesWithoutGeoref() : mpAtlas->GetCurrentMap()->GetAllKeyFrames(false);
    if(kfsWithoutGeoref.empty())
        return false;
    
    const auto georefKfs = mGeometricReferencer.getFramesForGeorefEstimation();
    Verbose::PrintMess("Georef function called with KFs :" + to_string(georefKfs.size()), Verbose::VERBOSITY_DEBUG);
    auto pose_scale_opt = mGeometricReferencer.apply(georefKfs, mbGeorefUpdate);
    if(pose_scale_opt.has_value()){
        const auto Tgw = pose_scale_opt.value();
        Verbose::PrintMess("Georef applied to KFs: " + to_string(kfsWithoutGeoref.size()), Verbose::VERBOSITY_DEBUG);
        for (const auto& pKF : kfsWithoutGeoref){
            const auto Twc = pKF->GetPoseInverse();
            const auto Tgc = mGeometricReferencer.getCurrentTransform()*Sophus::Sim3d(1.0,Twc.unit_quaternion().cast<double>(),Twc.translation().cast<double>());
            pKF->SetGNSSCameraPose(Tgc);
            auto vpMPs = pKF->GetMapPointMatches();
            for_each(execution::par, vpMPs.begin(), vpMPs.end(), [&](auto pMP)
            {
                if(pMP){
                    if(!pMP->isBad())                  
                        pMP->UpdateGNSSPos(mGeometricReferencer.getCurrentTransform());
                }
            });
        }
        mGeometricReferencer.updateGeorefKFsCount(kfsWithoutGeoref.size());
    }
    return pose_scale_opt.has_value();
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=30;
    const vector<shared_ptr<KeyFrame>> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<shared_ptr<KeyFrame>> vpTargetKFs;
    for(vector<shared_ptr<KeyFrame>>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        auto pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    const size_t coviseSize = 20;
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<shared_ptr<KeyFrame>> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(coviseSize);
        for(vector<shared_ptr<KeyFrame>>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            auto pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial)
    {
        auto pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<coviseSize && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    auto vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<shared_ptr<KeyFrame>>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        auto pKFi = *vit;
        matcher.Fuse(pKFi,vpMapPointMatches, 30.0, false);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<shared_ptr<MapPoint>> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<shared_ptr<KeyFrame>>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        auto pKFi = *vitKF;

        auto vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(auto vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            auto pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates, 10.0);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        auto pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateDepth();
            }
        }
    }

    // Update connections in covisibility graph
    unique_lock<mutex> lock(*getGlobalDataMutex());
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;

    mTElapsedTime = 0.f;
    mbNotBA2 = true;
    mbNotBA1 = true;
    mbBadImu=false;

    mbResetRequested = false;
    mGeometricReferencer.clear();

    ResetNewKeyFrames();
    Verbose::PrintMess("Local Mapping Release", Verbose::VERBOSITY_NORMAL);
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<shared_ptr<KeyFrame>> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        auto aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }

    for(vector<shared_ptr<KeyFrame>>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        auto pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const auto vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0; i<vpMapPoints.size(); i++)
        {
            auto pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel =  pKF->mvKeysUn->operator[](i).octave;
                        const map<shared_ptr<KeyFrame>, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<shared_ptr<KeyFrame>, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            auto pKFi = mit->first;
                            if(pKFi->mnId==pKF->mnId)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = pKFi->mvKeysUn->operator[](leftIndex).octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetInertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            else
            {
                pKF->SetBadFlag();
            }
        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

void LocalMapping::RequestReset()
{
    
    Verbose::PrintMess("LM: Map reset requested", Verbose::VERBOSITY_NORMAL);
    mbResetRequested = true;
}

void LocalMapping::ResetIfRequested()
{   
    if(mbResetRequested)
    {
        Verbose::PrintMess("LM: Reseting Atlas in Local Mapping...", Verbose::VERBOSITY_NORMAL);
        ResetNewKeyFrames();
        mlpRecentAddedMapPoints.clear();

        // Inertial parameters
        mTElapsedTime = 0.f;
        mbNotBA2 = true;
        mbNotBA1 = true;
        mbBadImu=false;
        mGeometricReferencer.clear();
        mbResetRequested = false;

        Verbose::PrintMess("LM: End reseting Local Mapping...", Verbose::VERBOSITY_NORMAL);
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}


bool LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA, int itsFIBA, float minTime, size_t nMinKF)
{
    if (mbResetRequested)
        return false;

    // if(mpAtlas->KeyFramesInMap()<nMinKF)
    //     return false;

    const auto kf_size = mpAtlas->GetCurrentMap()->GetAllKeyFrames(false).size();
    if(kf_size<nMinKF)
        return false;

    if(mTElapsedTime<minTime)
        return false;


    Verbose::PrintMess("Init IMU: Elapsed Time: " + to_string(mTElapsedTime) + " min KF: " + to_string(kf_size), Verbose::VERBOSITY_DEBUG);
    bInitializing = true;

    // We lock here so that no new kfs can be generated
    unique_lock<mutex> lockGlobal(*getGlobalDataMutex());
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();

    auto vpKF = mpAtlas->GetCurrentMap()->GetAllKeyFrames(true);
    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);


    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate); 

    // Compute and KF velocities mRw_gravity estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<shared_ptr<KeyFrame>>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;
            

            auto vel = (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Verbose::PrintMess("InitializeIMU - vel x: " + to_string(vel(0)) + " y: " + to_string(vel(1)) + " z: "+ to_string(vel(2)), Verbose::VERBOSITY_DEBUG);
            dirG -= (*itKF)->mPrevKF->GetImuRotation() * vel;
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }
        auto dirG_norm = dirG.norm();
        Verbose::PrintMess("InitializeIMU - dirG norm: " + to_string(dirG_norm), Verbose::VERBOSITY_DEBUG);
        // We dont have a 0 check here. If norm is 0 it should crash because there likely is a data problem
        dirG = dirG/dirG_norm;
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        Verbose::PrintMess("InitializeIMU - nv norm: " + to_string(nv), Verbose::VERBOSITY_DEBUG);
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Verbose::PrintMess("InitializeIMU - before exp call ...", Verbose::VERBOSITY_DEBUG);
        Eigen::Vector3f vzg = v*ang/nv;
        mRw_gravity = Sophus::SO3f::exp(vzg).matrix().cast<double>();
    }
    else
    {
        mRw_gravity = Eigen::Matrix3d::Identity();
        mb_gyro = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mb_accelerometer = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;
    {
        //unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate); 
        Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRw_gravity, mScale, mb_gyro, mb_accelerometer, mbFixScale, infoInertial, false, false, priorG, priorA);
    }

    if (mScale<1e-1)
    {
        Verbose::PrintMess("scale too small", Verbose::VERBOSITY_DEBUG);
        bInitializing=false;
        return false;
    }

    // Before this line we are not changing the map
    {
        //unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            Verbose::PrintMess("InitializeIMU - Scale Update", Verbose::VERBOSITY_DEBUG);
            const auto Ryw = Sophus::SO3d::fitToSO3(mRw_gravity.transpose());
            const Sophus::Sim3d Tw_gravity(mScale, Ryw.unit_quaternion(), Eigen::Vector3d::Zero());
            //const Sophus::Sim3d Tw_gravity(mScale, Ryw.unit_quaternion(), Eigen::Vector3d(1812328, 1812328, 1812328));
            //const Sophus::Sim3d Tw_gravity(mScale, Ryw.unit_quaternion(), Eigen::Vector3d(18123, 18123, 18123));
            const auto Tw_gravityf = Tw_gravity.cast<float>();
            UpdateTrackerAndMapCoordianateFrames(vpKF, Tw_gravityf, vpKF.front()->GetImuBias());
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < N; i++) {
                vpKF[i]->bImu = true;
            }
    }

    //mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
    if (!mpAtlas->isImuInitialized())
    {
        mpAtlas->SetImuInitialized();
        mpCurrentKeyFrame->bImu = true;
    }

    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
    if (bFIBA)
    {
        //unique_lock<mutex> lockGlobal(*getGlobalDataMutex());
        //unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate); 
        Verbose::PrintMess("Start Global Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), itsFIBA, false, 0, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), itsFIBA, false, 0, NULL, false, priorG, priorA);
    }

    chrono::steady_clock::time_point t5 = chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_DEBUG);

    // TODO: Investigate this set
    mpTracker->setTrackingState(Tracking::OK);
    bInitializing = false;

    // mpAtlas->GetCurrentMap()->IncreaseChangeIndex();

    return true;
}

void LocalMapping::UpdateTrackerAndMapCoordianateFrames(std::vector<shared_ptr<KeyFrame>> sortedKeyframes, const Sophus::Sim3f &Sim3_Tyw, const std::optional<IMU::Bias>& b_option){
    //TODO IMU Frame is hardcoded in pipeline as const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE); -> this should be aligned to some rotation frame if we are to support a GNSS alignged system
    mpTracker->UpdateInitialFrame(Sim3_Tyw);
    // const auto Sim3_Tiy = Sophus::Sim3f(1.0, mpTracker->GetInitialFrame()->GetPoseInverse().unit_quaternion(), mpTracker->GetInitialFrame()->GetPoseInverse().translation());
    // const auto Sim3_Tiw = Sim3_Tiy*Sim3_Tyw;
    // std::optional<IMU::Bias> b_temp = b_option;
    // if(b_option.has_value()){
    //     //b_temp.value().rotateBias(Sim3_Tiy.quaternion().normalized());
    //     for(auto kf : sortedKeyframes){
    //         kf->SetNewBias(b_temp.value());
    //         if (kf->mpImuPreintegrated)
    //             kf->mpImuPreintegrated->Reintegrate();
    //     }
    // }


    mpAtlas->GetCurrentMap()->UpdateKFsAndMapCoordianteFrames(sortedKeyframes, Sim3_Tyw, b_option);
    for(auto kf : sortedKeyframes){
        if (kf->mpImuPreintegrated)
            kf->mpImuPreintegrated->Reintegrate();
    }
    mpTracker->UpdateCoordinateFrames(Sim3_Tyw, b_option);


    // const auto georef_pose = getGeorefTransform();
    // // Coordiante Frames should be aligned, we only need to set the translation
    // const auto sortedKfs = mpAtlas->GetCurrentMap()->GetAllKeyFrames(true);
    // const auto currentFramePointer = mpTracker->GetCurrentFrame();
    // const auto initialFramePointer = mpTracker->GetInitialFrame();
    // const auto lastFramePointer = mpTracker->GetLastFrame();
    // const auto lastKFPointer = mpTracker->GetLastKeyFrame();
    // const auto firstKf = sortedKfs.front();
    // const auto lastKF = sortedKfs.back();


    // const auto deltaPose = initialFramePointer->GetPose()*lastKF->GetPoseInverse();
    // //const auto gnssDeltaGNSSPose = initialFramePointer->GetGNSSCameraPose().inverse()*lastKF->GetGNSSCameraPose();
    // const auto poseWorld = lastKF->GetPoseInverse();


    // const Eigen::Vector3f gnssDeltaTranslation = mpTracker->GetCurrentFrame()->GetGNSS() - mpTracker->GetInitialFrame()->GetGNSS();


    // Verbose::PrintMess("Transformation matrix:", Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess(to_string(georef_pose.rotationMatrix()(0,0)) + " " + to_string(georef_pose.rotationMatrix()(0,1)) + " " + to_string(georef_pose.rotationMatrix()(0,2)) + " " + to_string(georef_pose.translation()(0)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess(to_string(georef_pose.rotationMatrix()(1,0)) + " " + to_string(georef_pose.rotationMatrix()(1,1)) + " " + to_string(georef_pose.rotationMatrix()(1,2)) + " " + to_string(georef_pose.translation()(1)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess(to_string(georef_pose.rotationMatrix()(2,0)) + " " + to_string(georef_pose.rotationMatrix()(2,1)) + " " + to_string(georef_pose.rotationMatrix()(2,2)) + " " + to_string(georef_pose.translation()(2)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("Scale: " + to_string(georef_pose.scale()), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("\n", Verbose::VERBOSITY_NORMAL);


    // Verbose::PrintMess("TRACK: Current Poses\n ", Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("X: " +  to_string(currentFramePointer->GetPoseInverse().translation()(0)) + " Y: " + to_string(currentFramePointer->GetPoseInverse().translation()(1)) + " Z: " + to_string(currentFramePointer->GetPoseInverse().translation()(2)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("Latest KF X: " +  to_string(lastKFPointer->GetPoseInverse().translation()(0)) + " Y: " + to_string(lastKFPointer->GetPoseInverse().translation()(1)) + " Z: " + to_string(lastKFPointer->GetPoseInverse().translation()(2)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("Initial Frame X: " +  to_string(initialFramePointer->GetPoseInverse().translation()(0)) + " Y: " + to_string(initialFramePointer->GetPoseInverse().translation()(1)) + " Z: " + to_string(initialFramePointer->GetPoseInverse().translation()(2)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("Initial Frame Quat X: " +  to_string(initialFramePointer->GetPoseInverse().unit_quaternion().x()) + " Y: " + to_string(initialFramePointer->GetPoseInverse().unit_quaternion().y()) + " Z: " + to_string(initialFramePointer->GetPoseInverse().unit_quaternion().z()) + " W: " + to_string(initialFramePointer->GetPoseInverse().unit_quaternion().w()), Verbose::VERBOSITY_NORMAL);

    // Verbose::PrintMess("Delta KF: " +  to_string(deltaPose.translation()(0)) + " Y: " + to_string(deltaPose.translation()(1)) + " Z: " + to_string(deltaPose.translation()(2)), Verbose::VERBOSITY_NORMAL);
    // //Verbose::PrintMess("Delta GNSS KF: " +  to_string(gnssDeltaGNSSPose.translation()(0)) + " Y: " + to_string(gnssDeltaGNSSPose.translation()(1)) + " Z: " + to_string(gnssDeltaGNSSPose.translation()(2)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("Pose KF: " +  to_string(poseWorld.translation()(0)) + " Y: " + to_string(poseWorld.translation()(1)) + " Z: " + to_string(poseWorld.translation()(2)), Verbose::VERBOSITY_NORMAL);
    // Verbose::PrintMess("\n", Verbose::VERBOSITY_NORMAL);

    //throw std::runtime_error("GNSS fallback not implemented yet");

}

bool LocalMapping::IsInitializing() const
{
    return bInitializing;
}

double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

shared_ptr<KeyFrame> LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

// This can become a deadlock if trying to aquire during reset, since the tracker is waiting to the mapper.
shared_ptr<mutex> LocalMapping::getGlobalDataMutex(){
    return mMutexPtrGlobalData;
}

void LocalMapping::setLatestOptimizedKFPoses (const vector<pair<long unsigned int,Sophus::SE3f>>& optimizedKFPoses){
    unique_lock<std::mutex> lock(mMutexLatestOptimizedKFPoses);
    mLatestOptimizedKFPoses = optimizedKFPoses;
}

vector<pair<long unsigned int,Sophus::SE3f>> LocalMapping::getLatestOptimizedKFPoses() {
    unique_lock<std::mutex> lock(mMutexLatestOptimizedKFPoses);
    return mLatestOptimizedKFPoses;
}

bool LocalMapping::isGeorefInitialized() const {
    return !mbUseGNSS || mGeometricReferencer.isInitialized();
}

Sophus::Sim3d LocalMapping::getGeorefTransform() {
    return mGeometricReferencer.getCurrentTransform();
}

} //namespace ORB_SLAM
