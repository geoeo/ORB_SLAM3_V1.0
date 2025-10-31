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


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>
#include <utility>
#include <tuple>
#include <opencv2/core/core.hpp>

#include <Tracking.h>
#include <FrameDrawer.h>
#include <MapDrawer.h>
#include <Atlas.h>
#include <LocalMapping.h>
#include <LoopClosing.h>
#include <KeyFrameDatabase.h>
#include <ORBVocabulary.h>
#include <Viewer.h>
#include <ImuTypes.h>
#include <Settings.h>
#include <Verbose.h>
#include <KeyPoint.h>


namespace ORB_SLAM3
{

class FrameDrawer;
class MapDrawer;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };

    // File type
    enum FileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Headless setup
    System(const std::string &strVocFile, const CameraParameters &cam_settings, const ImuParameters &imu_settings, const OrbParameters &orb_settings, const LocalMapperParameters &local_mapper_settings,
        const TrackerParameters& tracker_settings, const eSensor sensor, bool activeLC, bool bUseViewer);

    
    ~System();

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    std::tuple<Sophus::SE3f, bool,bool, unsigned long int, vector<float>> TrackMonocular(const cv::cuda::HostMem &im_managed, const double &timestamp, const std::vector<IMU::Point>& vImuMeas = std::vector<IMU::Point>(), bool hasGNSS=false, Eigen::Vector3f GNSSPosition=Eigen::Vector3f::Zero(), std::string filename="");


    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();
    bool isShutDown();

    unsigned int GetLastKeyFrameId();

    // Returns image of the current frame being tracked with features
    cv::Mat DrawTrackedImage();

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetActiveReferenceMapPoints(); 
    std::shared_ptr<std::vector<KeyPoint>> GetTrackedKeyPointsUn();
    std::vector<KeyFrame*> GetAllKeyframes();
    std::shared_ptr<std::mutex> getGlobalDataMutex();

    // For debugging
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    bool isGeoreferenced() const;
    bool isImuInitialized() const;
    void setGeoreference(bool is_georeferenced);
    std::vector<std::pair<long unsigned int,Sophus::SE3f>> getLatestOptimizedKFPoses();


private:
    bool has_suffix(const std::string &str, const std::string &suffix);
    std::unique_lock<std::mutex> scoped_mutex_lock(std::mutex &m);

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Shutdown flag
    bool mbShutDown;

    // Tracking state
    int mTrackingState;
    std::shared_ptr<std::vector<KeyPoint>> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    //
    std::string mStrLoadAtlasFromFile;
    std::string mStrSaveAtlasToFile;

    std::string mStrVocabularyFilePath;

    Settings* settings_;
};

}// namespace ORB_SLAM

