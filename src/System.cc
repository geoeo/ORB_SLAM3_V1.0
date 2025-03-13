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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <openssl/md5.h>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <tracy.hpp>

using namespace std;
using namespace cuda_cv_managed_memory;

auto scoped_mutex_lock(std::mutex &m){
    ZoneScopedC(tracy::Color::Coral);
    return std::unique_lock(m);
}

namespace ORB_SLAM3
{

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

bool System::has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}

System::System(const std::string &strVocFile, const CameraParameters &cam, const ImuParameters &imu, const OrbParameters &orb, 
    const eSensor sensor, int frameGridCols, int frameGridRows ,bool activeLC, bool bUseViewer):
    mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false), mbResetActiveMap(false),
    mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false)
{
  // Output welcome message
  cout << endl <<
       "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
       "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
       "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
       "This is free software, and you are welcome to redistribute it" << endl <<
       "under certain conditions. See LICENSE.txt." << endl << endl;

  cout << "Input sensor was set to: ";

  if(mSensor==MONOCULAR)
    cout << "Monocular" << endl;
  else if(mSensor==STEREO)
    cout << "Stereo" << endl;
  else if(mSensor==RGBD)
    cout << "RGB-D" << endl;
  else if(mSensor==IMU_MONOCULAR)
    cout << "Monocular-Inertial" << endl;
  else if(mSensor==IMU_STEREO)
    cout << "Stereo-Inertial" << endl;

  //Load ORB Vocabulary
  cout << endl << "Loading ORB Vocabulary from " << strVocFile << endl;

  mpVocabulary = new ORB_SLAM3::ORBVocabulary();
  bool bVocLoad = false;
  // chose loading method based on file extension
  if (has_suffix(strVocFile, ".txt"))
    bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  else
    bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);

  if(!bVocLoad)
  {
    cerr << "Wrong path to vocabulary. " << endl;
    cerr << "Falied to open at: " << strVocFile << endl;
    exit(-1);
  }
  cout << "Vocabulary loaded!" << endl << endl;

  //Create KeyFrame Database
  mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

  //Create the Atlas
  //mpMap = new Map();
  mpAtlas = new Atlas(0);
  //----

  if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR)
    mpAtlas->SetInertialSensor();

  settings_ = new Settings(cam, imu, orb,mSensor, frameGridCols, frameGridRows);
  cout << (*settings_) << endl;

    mpFrameDrawer = nullptr;
    mpMapDrawer = nullptr;
    //if(bUseViewer){
        //Create Drawers. These are used by the Viewer
        mpFrameDrawer = new FrameDrawer(mpAtlas);
        mpMapDrawer = new MapDrawer(mpAtlas, std::string(), settings_);
    //}



  //Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this constructor)

    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                                mpAtlas, mpKeyFrameDatabase, std::string(), mSensor, settings_);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor==MONOCULAR || mSensor==IMU_MONOCULAR,
                                     mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD, std::string());
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run,mpLocalMapper);
    mpLocalMapper->mInitFr = 0; // seems to be ununsed
    if(settings_)
        mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    if(mpLocalMapper->mThFarPoints!=0)
    {
        cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;

    //Initialize the Loop Closing thread and launch
    // mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR
    //mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR, activeLC); // mSensor!=MONOCULAR);
    //mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    //mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    //mpLocalMapper->SetLoopCloser(mpLoopCloser);

    //mpLoopCloser->SetTracker(mpTracker);
    //mpLoopCloser->SetLocalMapper(mpLocalMapper);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,std::string(),settings_);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        //mpLoopCloser->mpViewer = mpViewer;
        mpViewer->both = mpFrameDrawer->both;
    }

    // Fix verbosity
    Verbose::SetTh(Verbose::VERBOSITY_NORMAL);
}

System::~System(){
    delete mpVocabulary;
    delete mpKeyFrameDatabase;
    delete mpAtlas;
    delete settings_;
    delete mpTracker;
    delete mpLocalMapper;

    mptLocalMapping->join();
    delete mpLocalMapper;

    if(mptLoopClosing){
        mptLocalMapping->join();
        delete mptLoopClosing;
    }


    if(mpViewer){
        mptViewer->join();
        delete mpViewer;
    }
}

tuple<Sophus::SE3f, bool,bool, unsigned long int, vector<float>> System::TrackMonocular(const cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr &im_managed, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{

    ZoneNamedN(TrackMonocular, "TrackMonocular", true);  // NOLINT: Profiler

    {
        //unique_lock<mutex> lock(mMutexReset);
        auto lock = scoped_mutex_lock( mMutexReset );

        if(mbShutDown)
            return {Sophus::SE3f(),false,false,0, {}};
    }

    if(mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial." << endl;
        exit(-1);
    }


    // Check mode change
    {
        //unique_lock<mutex> lock(mMutexMode);
        auto lock = scoped_mutex_lock( mMutexMode );
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        //unique_lock<mutex> lock(mMutexReset);
        auto lock = scoped_mutex_lock( mMutexReset );
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            cout << "SYSTEM-> Reseting active map in monocular case" << endl;
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    auto [Tcw,id, isKeyframe] = mpTracker->GrabImageMonocular(im_managed,timestamp,filename);

    //unique_lock<mutex> lock2(mMutexState);
    auto lock = scoped_mutex_lock( mMutexState );
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    auto isBAComplete = mpTracker->isBACompleteForMap();
    auto computedScales = mpTracker->getMapScales();

    return {Tcw,isBAComplete,isKeyframe,id, computedScales};
}



void System::ActivateLocalizationMode()
{
    //unique_lock<mutex> lock(mMutexMode);
    auto lock = scoped_mutex_lock( mMutexMode );
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    //unique_lock<mutex> lock(mMutexMode);
    auto lock = scoped_mutex_lock( mMutexMode );
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpAtlas->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    //unique_lock<mutex> lock(mMutexReset);
    auto lock = scoped_mutex_lock( mMutexReset );
    mbReset = true;
}

void System::ResetActiveMap()
{
    //unique_lock<mutex> lock(mMutexReset);
    auto lock = scoped_mutex_lock( mMutexReset );
    mbResetActiveMap = true;
}

void System::Shutdown()
{
    {
        //unique_lock<mutex> lock(mMutexReset);
        auto lock = scoped_mutex_lock( mMutexReset );
        mbShutDown = true;
    }

    cout << "Shutdown" << endl;

    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    /*if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }*/

    // Wait until all thread have effectively stopped
    /*while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        if(!mpLocalMapper->isFinished())
            cout << "mpLocalMapper is not finished" << endl;*/
        /*if(!mpLoopCloser->isFinished())
            cout << "mpLoopCloser is not finished" << endl;
        if(mpLoopCloser->isRunningGBA()){
            cout << "mpLoopCloser is running GBA" << endl;
            cout << "break anyway..." << endl;
            break;
        }*/
        /*usleep(5000);
    }*/

    if(!mStrSaveAtlasToFile.empty())
    {
        Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile, Verbose::VERBOSITY_NORMAL);
        SaveAtlas(FileType::BINARY_FILE);
    }

    /*if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");*/

#ifdef REGISTER_TIMES
    mpTracker->PrintTimeStats();
#endif
}

bool System::isShutDown() {
    //unique_lock<mutex> lock(mMutexReset);
    auto lock = scoped_mutex_lock( mMutexReset );
    return mbShutDown;
}

//This is not really correct -> remove in future use bool value from tracking
unsigned int System::GetLastKeyFrameId()
{
  //unique_lock<mutex> lock(mMutexState);
  auto lock = scoped_mutex_lock( mMutexState );
  return mpTracker->GetLastKeyFrameId();
}

cv::Mat System::DrawTrackedImage()
{
  //unique_lock<mutex> lock(mMutexState);
  auto lock = scoped_mutex_lock( mMutexState );
  return mpFrameDrawer->DrawFrame();
}


int System::GetTrackingState()
{
    //unique_lock<mutex> lock(mMutexState);
    auto lock = scoped_mutex_lock( mMutexState );
    return mTrackingState;
}

vector<MapPoint*> System::GetActiveReferenceMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    return pActiveMap->GetReferenceMapPoints();
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    //unique_lock<mutex> lock(mMutexState);
    auto lock = scoped_mutex_lock( mMutexState );
    return mTrackedKeyPointsUn;
}


std::vector<KeyFrame*> System::GetAllKeyframes() {
    return mpAtlas->GetAllKeyFrames();
}

double System::GetTimeFromIMUInit()
{
    double aux = mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    if ((aux>0.) && mpAtlas->isImuInitialized())
        return mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    else
        return 0.f;
}

bool System::isLost()
{
    if (!mpAtlas->isImuInitialized())
        return false;
    else
    {
        if ((mpTracker->mState==Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
            return true;
        else
            return false;
    }
}


bool System::isFinished()
{
    return (GetTimeFromIMUInit()>0.1);
}

void System::ChangeDataset()
{
    if(mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12)
    {
        mpTracker->ResetActiveMap();
    }
    else
    {
        mpTracker->CreateMapInAtlas();
    }

    mpTracker->NewDataset();
}

float System::GetImageScale()
{
    return mpTracker->GetImageScale();
}

bool System::isGeoreferenced() {
    return mpTracker->isGeoreferenced();
}

void System::setGeoreference(bool is_georeferenced){
    mpTracker->setGeoreference(is_georeferenced);
}

void System::SaveAtlas(int type){
    if(!mStrSaveAtlasToFile.empty())
    {
        // Save the current session
        mpAtlas->PreSave();

        string pathSaveFileName = "./";
        pathSaveFileName = pathSaveFileName.append(mStrSaveAtlasToFile);
        pathSaveFileName = pathSaveFileName.append(".osa");

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found+1);

        if(type == TEXT_FILE) // File text
        {
            cout << "Starting to write the save text file " << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            cout << "End to write the save text file" << endl;
        }
        else if(type == BINARY_FILE) // File binary
        {
            cout << "Starting to write the save binary file" << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            cout << "End to write save binary file" << endl;
        }
    }
}

bool System::LoadAtlas(int type)
{
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    string pathLoadFileName = "./";
    pathLoadFileName = pathLoadFileName.append(mStrLoadAtlasFromFile);
    pathLoadFileName = pathLoadFileName.append(".osa");

    if(type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save text file " << endl;
        isRead = true;
    }
    else if(type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file"  << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if(isRead)
    {
        //Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);

        if(strInputVocabularyChecksum.compare(strVocChecksum) != 0)
        {
            cout << "The vocabulary load isn't the same which the load session was created " << endl;
            cout << "-Vocabulary name: " << strFileVoc << endl;
            return false; // Both are differents
        }

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetORBVocabulary(mpVocabulary);
        mpAtlas->PostLoad();

        return true;
    }
    return false;
}

string System::CalculateCheckSum(string filename, int type)
{
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if(type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if ( !f.is_open() )
    {
        cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init (&md5Context);
    while ( int count = f.readsome(buffer, sizeof(buffer)))
    {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context );

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        char aux[10];
        sprintf(aux,"%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}

} //namespace ORB_SLAM

