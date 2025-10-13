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

#include <KeyFrame.h>
#include <Atlas.h>
#include <LoopClosing.h>
#include <Tracking.h>
#include <KeyFrameDatabase.h>
#include <Settings.h>
#include <System.h>
#include <GeometricReferencer.hpp>

#include <mutex>
#include <atomic>
#include <vector>
#include <utility>


namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;

class LocalMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalMapping(System* pSys, Atlas* pAtlas, const float bMonocular, bool bInertial, const LocalMapperParameters &local_mapper);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue();

    bool IsInitializing() const;
    double GetCurrKFTime();
    KeyFrame* GetCurrKF();

    std::shared_ptr<std::mutex> getKeyFrameChangeMutex();
    std::shared_ptr<std::mutex> getGlobalDataMutex();
    void setLatestOptimizedKFPoses (const std::vector<std::pair<long unsigned int,Sophus::SE3f>>& optimizedKFPoses);
    std::vector<std::pair<long unsigned int,Sophus::SE3f>> getLatestOptimizedKFPoses();

    Eigen::MatrixXd mcovInertial;
    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    double mInitTime;
    double mCostTime;

    unsigned int mInitSect;
    double mFirstTs;
    int mnMatchesInliers;

    // For debugging (erase in normal mode)
    int mInitFr;
    int mIdxIteration;
    std::string strSequence;

    bool mbNotBA1;
    bool mbNotBA2;
    bool mbBadImu;

    bool mbWriteStats;

    // not consider far points (clouds)
    bool mbFarPoints;
    float mThFarPoints;

protected:

    bool CheckNewKeyFrames();
    void ResetNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();
    bool GeoreferenceKeyframes();

    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    System *mpSystem;

    bool mbMonocular;
    bool mbInertial;

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetRequestedActiveMap;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;
    std::mutex mMutexImuInit;
    std::mutex mMutexLatestOptimizedKFPoses;


    bool mbAbortBA;
    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    std::shared_ptr<std::mutex> mMutexPtrGlobalData;

    bool InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFIBA = false, int itsFIBA = 200, float minTime = 5, size_t nMinKF = 10);
    void ScaleRefinement();

    std::atomic_bool bInitializing;

    Eigen::MatrixXd infoInertial;
    int mNumLM;
    int mNumKFCulling;

    float mTElapsedTime;

    int countRefinement;

    const float resetTimeThresh;
    const float minTimeForImuInit;
    const float minTimeForVIBA1;
    const float minTimeForVIBA2;
    const float minTimeForFullBA;
    const float itsFIBAInit;
    const float itsFIBA1;
    int writeKFAfterGeorefCount;
    int writeKFAfterGBACount;

    bool mbUseGNSS;
    bool mbUseGNSSBA;
    bool mbWriteGNSSData;

    std::vector<std::pair<long unsigned int,Sophus::SE3f>> mLatestOptimizedKFPoses;
    GeometricReferencer mGeometricReferencer;



    };

} //namespace ORB_SLAM3

