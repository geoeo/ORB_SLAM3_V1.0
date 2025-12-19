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

#include "Atlas.h"
#include "Viewer.h"

#include <thread>
#include "CameraModels/GeometricCamera.h"
#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"

using namespace std;

namespace ORB_SLAM3
{

Atlas::Atlas(){
    mpCurrentMap = nullptr;
}

Atlas::Atlas(int initKFid): mnLastInitKFidMap(initKFid), mHasViewer(false)
{
    mpCurrentMap = nullptr;
    CreateNewMap();
}

Atlas::~Atlas()
{

}

void Atlas::CreateNewMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    Verbose::PrintMess("Creation of new map with id: " + to_string(Map::nNextId), Verbose::VERBOSITY_DEBUG);
    if(mpCurrentMap){
        if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid()+1; //The init KF is the next of current maximum

        mpCurrentMap->SetStoredMap();
        Verbose::PrintMess("Stored map with ID: " + to_string(mpCurrentMap->GetId()), Verbose::VERBOSITY_DEBUG);
    }
    Verbose::PrintMess("Creation of new map with last KF id: " + to_string(mnLastInitKFidMap), Verbose::VERBOSITY_DEBUG);

    mpCurrentMap = make_shared<Map>(mnLastInitKFidMap);
    mpCurrentMap->SetCurrentMap();
    mspMaps.insert(mpCurrentMap);
}

void Atlas::ChangeMap(std::shared_ptr<Map> pMap)
{
    unique_lock<mutex> lock(mMutexAtlas);
    Verbose::PrintMess("Change to map with id: " + to_string(pMap->GetId()), Verbose::VERBOSITY_NORMAL);
    if(mpCurrentMap){
        mpCurrentMap->SetStoredMap();
    }

    mpCurrentMap = pMap;
    mpCurrentMap->SetCurrentMap();
}

unsigned long int Atlas::GetLastInitKFid()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mnLastInitKFidMap;
}

void Atlas::SetViewer(Viewer* pViewer)
{
    mpViewer = pViewer;
    mHasViewer = true;
}

void Atlas::AddKeyFrame(shared_ptr<KeyFrame> pKF)
{
    auto pMapKF = pKF->GetMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddMapPoint(std::shared_ptr<MapPoint> pMP)
{
    auto pMapMP = pMP->GetMap();
    pMapMP->AddMapPoint(pMP);
}

GeometricCamera* Atlas::AddCamera(GeometricCamera* pCam)
{
    //Check if the camera already exists
    bool bAlreadyInMap = false;
    int index_cam = -1;
    for(size_t i=0; i < mvpCameras.size(); ++i)
    {
        GeometricCamera* pCam_i = mvpCameras[i];
        if(!pCam) Verbose::PrintMess("Not pCam", Verbose::VERBOSITY_DEBUG);
        if(!pCam_i) Verbose::PrintMess("Not pCam_i", Verbose::VERBOSITY_DEBUG);
        if(pCam->GetType() != pCam_i->GetType())
            continue;

        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            if(((Pinhole*)pCam_i)->IsEqual(pCam))
            {
                bAlreadyInMap = true;
                index_cam = i;
            }
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            if(((KannalaBrandt8*)pCam_i)->IsEqual(pCam))
            {
                bAlreadyInMap = true;
                index_cam = i;
            }
        }
    }

    if(bAlreadyInMap)
    {
        return mvpCameras[index_cam];
    }
    else{
        mvpCameras.push_back(pCam);
        return pCam;
    }
}

std::vector<GeometricCamera*> Atlas::GetAllCameras()
{
    return mvpCameras;
}

void Atlas::SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &vpMPs)
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetReferenceMapPoints(vpMPs);
}

void Atlas::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->InformNewBigChange();
}

int Atlas::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetLastBigChangeIdx();
}

long unsigned int Atlas::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->MapPointsInMap();
}

long unsigned Atlas::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->KeyFramesInMap();
}

vector<shared_ptr<KeyFrame>> Atlas::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllKeyFrames(false);
}

std::vector<std::shared_ptr<MapPoint>> Atlas::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllMapPoints();
}

std::vector<std::shared_ptr<MapPoint>> Atlas::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetReferenceMapPoints();
}

bool Atlas::isBACompleteForMap() {
    unique_lock<mutex> lock(mMutexAtlas);
    auto complete = false;

    if(mpCurrentMap)
        complete = mpCurrentMap->GetInertialFullBA();

    return complete;   
}

vector<float> Atlas::getMapScales() {
    unique_lock<mutex> lock(mMutexAtlas);
    vector<float> scales = {};

    if(mpCurrentMap)
        scales = mpCurrentMap->getVIBAScales();
    
    return scales;   
}

vector<shared_ptr<Map>> Atlas::GetAllMaps()
{
    unique_lock<mutex> lock(mMutexAtlas);
    struct compFunctor
    {
        inline bool operator()(shared_ptr<Map> elem1 ,shared_ptr<Map> elem2)
        {
            return elem1->GetId() < elem2->GetId();
        }
    };
    vector<shared_ptr<Map>> vMaps(mspMaps.begin(),mspMaps.end());
    sort(vMaps.begin(), vMaps.end(), compFunctor());
    return vMaps;
}

int Atlas::CountMaps()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mspMaps.size();
}

void Atlas::clearMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->clear();
}

void Atlas::clearAtlas()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mspMaps.clear();
    mpCurrentMap = nullptr;
    mnLastInitKFidMap = 0;
}

shared_ptr<Map> Atlas::GetCurrentMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    if(!mpCurrentMap)
        CreateNewMap();
    while(mpCurrentMap->IsBad()){
        this_thread::sleep_for(chrono::microseconds(3000));
        Verbose::PrintMess("Waiting for a valid current map", Verbose::VERBOSITY_NORMAL);
    }


    return mpCurrentMap;
}

void Atlas::SetMapBad(shared_ptr<Map> pMap)
{
    mspMaps.erase(pMap);
    pMap->SetBad();

    mspBadMaps.insert(pMap);
}

void Atlas::RemoveBadMaps()
{
    mspBadMaps.clear();
}

bool Atlas::isInertial()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return mpCurrentMap->IsInertial();
}

void Atlas::SetInertialSensor()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetInertialSensor();
}

void Atlas::SetImuInitialized()
{
    unique_lock<mutex> lock(mMutexAtlas);
    mpCurrentMap->SetImuInitialized();
}

bool Atlas::isImuInitialized()
{
    unique_lock<mutex> lock(mMutexAtlas);
    return !mpCurrentMap ? false : mpCurrentMap->isImuInitialized();
}

void Atlas::SetKeyFrameDatabase(std::shared_ptr<KeyFrameDatabase> pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

std::shared_ptr<KeyFrameDatabase> Atlas::GetKeyFrameDatabase()
{
    return mpKeyFrameDB;
}

std::shared_ptr<ORBVocabulary> Atlas::GetORBVocabulary()
{
    return mpORBVocabulary;
}

long unsigned int Atlas::GetNumLivedKF()
{
    unique_lock<mutex> lock(mMutexAtlas);
    long unsigned int num = 0;
    for(auto pMap_i : mspMaps)
    {
        num += pMap_i->GetAllKeyFrames(false).size();
    }

    return num;
}

long unsigned int Atlas::GetNumLivedMP() {
    unique_lock<mutex> lock(mMutexAtlas);
    long unsigned int num = 0;
    for (auto pMap_i : mspMaps) {
        num += pMap_i->GetAllMapPoints().size();
    }

    return num;
}

map<long unsigned int, shared_ptr<KeyFrame>> Atlas::GetAtlasKeyframes()
{
    map<long unsigned int, shared_ptr<KeyFrame>> mpIdKFs;
    for(auto pMap_i : mvpBackupMaps)
    {
        vector<shared_ptr<KeyFrame>> vpKFs_Mi = pMap_i->GetAllKeyFrames(false);

        for(shared_ptr<KeyFrame> pKF_j_Mi : vpKFs_Mi)
        {
            mpIdKFs[pKF_j_Mi->mnId] = pKF_j_Mi;
        }
    }

    return mpIdKFs;
}

} //namespace ORB_SLAM3
