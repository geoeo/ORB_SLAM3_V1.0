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

#include <MapPoint.h>
#include <KeyFrame.h>

#include <set>
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM3
{


//TODO: forward
class MapPoint;
class KeyFrame;
class Atlas;
class KeyFrameDatabase;

class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Map();
    Map(int initKFid);
    ~Map();

    void AddKeyFrame(std::shared_ptr<KeyFrame> pKF);
    void AddMapPoint(std::shared_ptr<MapPoint> pMP);
    void EraseMapPoint(std::shared_ptr<MapPoint> pMP);
    void EraseKeyFrame(shared_ptr<KeyFrame> pKF);
    void SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyFrames(bool sort);
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
    std::vector<std::shared_ptr<MapPoint>> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetId();

    long unsigned int GetInitKFid();
    void SetInitKFid(long unsigned int initKFif);
    long unsigned int GetMaxKFid();

    shared_ptr<KeyFrame> GetOriginKF();

    void SetCurrentMap();
    void SetStoredMap();

    bool HasThumbnail();
    bool IsInUse();

    void SetBad();
    bool IsBad();

    void clear();

    int GetMapChangeIndex();
    void IncreaseChangeIndex();
    int GetLastMapChange();
    void SetLastMapChange(int currentChangeId);

    void SetImuInitialized();
    bool isImuInitialized();

    void UpdateKFsAndMapCoordianteFrames(std::vector<shared_ptr<KeyFrame>> sortedKeyframes, const Sophus::Sim3f &Sim3_Tyw, const std::optional<IMU::Bias> &b_option);
    void ApplyGNSSTransformation(const Sophus::SE3f &T);

    void SetInertialSensor();
    bool IsInertial();
    void SetInertialBA1();
    void SetInertialBA2();
    void SetInertialFullBA();
    bool GetInertialBA1();
    bool GetInertialBA2();
    bool GetInertialFullBA();
    std::vector<float> getVIBAScales();

    void PrintEssentialGraph();
    bool CheckEssentialGraph();
    void ChangeId(long unsigned int nId);

    unsigned int GetLowerKFID();

    void printReprojectionError(std::list<std::shared_ptr<KeyFrame>> &lpLocalWindowKFs, std::shared_ptr<KeyFrame> mpCurrentKF, std::string &name, std::string &name_folder);
    static void writeKeyframesCsv(const std::string& path, const std::vector<std::shared_ptr<KeyFrame>> keyframes, char sep = ',', int precision = 17);
    static void writeKeyframesReprojectionErrors(const std::string& filename, const std::vector<std::shared_ptr<KeyFrame>> keyframes, char sep = ',', int precision = 17);

    std::vector<std::shared_ptr<KeyFrame>> mvpKeyFrameOrigins;
    std::vector<unsigned long int> mvBackupKeyFrameOriginsId;
    std::shared_ptr<KeyFrame> mpFirstRegionKF;
    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    bool mbFail;

    static long unsigned int nNextId;

    // DEBUG: show KFs which are used in LBA
    std::set<long unsigned int> msOptKFs;
    std::set<long unsigned int> msFixedKFs;

protected:

    long unsigned int mnId;

    std::set<std::shared_ptr<MapPoint>> mspMapPoints;
    std::set<std::shared_ptr<KeyFrame>> mspKeyFrames;

    // Save/load, the set structure is broken in libboost 1.58 for ubuntu 16.04, a vector is serializated
    std::vector<std::shared_ptr<MapPoint>> mvpBackupMapPoints;
    std::vector<std::shared_ptr<KeyFrame>> mvpBackupKeyFrames;

    std::shared_ptr<KeyFrame> mpKFinitial;
    std::shared_ptr<KeyFrame> mpKFlowerID;

    unsigned long int mnBackupKFinitialID;
    unsigned long int mnBackupKFlowerID;

    std::vector<std::shared_ptr<MapPoint>> mvpReferenceMapPoints;

    bool mbImuInitialized;

    int mnMapChange;
    int mnMapChangeNotified;

    long unsigned int mnInitKFid;
    long unsigned int mnMaxKFid;
    //long unsigned int mnLastLoopKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    bool mIsInUse;
    bool mbBad = false;

    bool mbIsInertial;
    bool mbIMU_BA1;
    bool mbIMU_BA2;
    bool mbIMU_FullBA;
    float mfScale;
    std::vector<float> mfScales;

    // Mutex
    std::mutex mMutexMap;

};

} //namespace ORB_SLAM3

