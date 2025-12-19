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

#include <Map.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <CameraModels/GeometricCamera.h>
#include <CameraModels/Pinhole.h>
#include <CameraModels/KannalaBrandt8.h>

#include <set>
#include <mutex>
#include <memory>

namespace ORB_SLAM3
{
//TODO: Forward
class Viewer;
class Map;
class MapPoint;
class KeyFrame;
class KeyFrameDatabase;
class Frame;

class Atlas
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Atlas();
    Atlas(int initKFid); // When its initialization the first map is created
    ~Atlas();

    void CreateNewMap();
    void ChangeMap(Map* pMap);

    unsigned long int GetLastInitKFid();

    void SetViewer(Viewer* pViewer);

    // Method for change components in the current map
    void AddKeyFrame(std::shared_ptr<KeyFrame> pKF);
    void AddMapPoint(std::shared_ptr<MapPoint> pMP);
    //void EraseMapPoint(MapPoint* pMP);
    //void EraseKeyFrame(KeyFrame* pKF);

    GeometricCamera* AddCamera(GeometricCamera* pCam);
    std::vector<GeometricCamera*> GetAllCameras();

    /* All methods without Map pointer work on current map */
    void SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    long unsigned int MapPointsInMap();
    long unsigned KeyFramesInMap();

    // Method for get data in current map
    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
    std::vector<std::shared_ptr<MapPoint>> GetReferenceMapPoints();
    bool isBACompleteForMap();
    std::vector<float> getMapScales();

    std::vector<Map*> GetAllMaps();

    int CountMaps();

    void clearMap();

    void clearAtlas();

    Map* GetCurrentMap();

    void SetMapBad(Map* pMap);
    void RemoveBadMaps();

    bool isInertial();
    void SetInertialSensor();
    void SetImuInitialized();
    bool isImuInitialized();

    std::map<long unsigned int, std::shared_ptr<KeyFrame>> GetAtlasKeyframes();

    void SetKeyFrameDatabase(std::shared_ptr<KeyFrameDatabase> pKFDB);
    std::shared_ptr<KeyFrameDatabase> GetKeyFrameDatabase();

    std::shared_ptr<ORBVocabulary> GetORBVocabulary();

    long unsigned int GetNumLivedKF();
    long unsigned int GetNumLivedMP();

protected:

    std::set<Map*> mspMaps;
    std::set<Map*> mspBadMaps;
    // Its necessary change the container from set to vector because libboost 1.58 and Ubuntu 16.04 have an error with this cointainer
    std::vector<Map*> mvpBackupMaps;

    Map* mpCurrentMap;

    std::vector<GeometricCamera*> mvpCameras;

    unsigned long int mnLastInitKFidMap;

    Viewer* mpViewer;
    bool mHasViewer;

    // Class references for the map reconstruction from the save file
    std::shared_ptr<KeyFrameDatabase> mpKeyFrameDB;
    std::shared_ptr<ORBVocabulary> mpORBVocabulary;

    // Mutex
    std::mutex mMutexAtlas;


}; // class Atlas

} // namespace ORB_SLAM3
