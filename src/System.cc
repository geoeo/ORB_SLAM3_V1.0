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



#include <System.h>
#include <Converter.h>
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <openssl/md5.h>
#include <tracy.hpp>
#include <thread>

using namespace std;

namespace ORB_SLAM3
{

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

std::unique_lock<std::mutex> System::scoped_mutex_lock(std::mutex &m){
    ZoneScopedC(tracy::Color::Coral);
    return std::unique_lock(m);
}

bool System::has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}

System::System(const std::string &strVocFile, const CameraParameters &cam_settings, const ImuParameters &imu_settings, const OrbParameters &orb_settings, const LocalMapperParameters &local_mapper_settings,
    const eSensor sensor, int frameGridCols, int frameGridRows ,bool activeLC, bool bUseViewer):
    mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false), mbResetActiveMap(false),
    mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false)
{


//TODO: Use Verbose struct
  // Output welcome message
  Verbose::PrintMess("ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." , Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("This program comes with ABSOLUTELY NO WARRANTY", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("This is free software, and you are welcome to redistribute it", Verbose::VERBOSITY_NORMAL);
  Verbose::PrintMess("under certain conditions. See LICENSE.txt.", Verbose::VERBOSITY_NORMAL);

  if(mSensor==MONOCULAR)
    Verbose::PrintMess("Monocular", Verbose::VERBOSITY_NORMAL);
  else if(mSensor==STEREO)
    Verbose::PrintMess("Stereo", Verbose::VERBOSITY_NORMAL);
  else if(mSensor==RGBD)
    Verbose::PrintMess("RGB-D", Verbose::VERBOSITY_NORMAL);
  else if(mSensor==IMU_MONOCULAR)
  Verbose::PrintMess("Monocular-Inertial", Verbose::VERBOSITY_NORMAL);
  else if(mSensor==IMU_STEREO)
    Verbose::PrintMess("Stereo-Inertial", Verbose::VERBOSITY_NORMAL);

  //Load ORB Vocabulary
  Verbose::PrintMess("Loading ORB Vocabulary from " + strVocFile, Verbose::VERBOSITY_NORMAL);

  mpVocabulary = new ORB_SLAM3::ORBVocabulary();
  bool bVocLoad = false;
  // chose loading method based on file extension
  if (has_suffix(strVocFile, ".txt"))
    bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  else
    bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);

  if(!bVocLoad)
  {
    Verbose::PrintMess("Error: Wrong path to vocabulary.", Verbose::VERBOSITY_NORMAL);
    Verbose::PrintMess("Failed to open at: " + strVocFile, Verbose::VERBOSITY_NORMAL);
    exit(-1);
  }
  cout << "Vocabulary loaded!" << endl << endl;

  //Create KeyFrame Database
  mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

  //Create the Atlas
    mpAtlas = new Atlas(0);

    if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR)
        mpAtlas->SetInertialSensor();

    settings_ = new Settings(cam_settings, imu_settings, orb_settings, mSensor, frameGridCols, frameGridRows);

    cout << (*settings_) << endl;

    mpFrameDrawer = new FrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, std::string(), settings_);
    

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)


    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor==MONOCULAR || mSensor==IMU_MONOCULAR,
                                     mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD, local_mapper_settings);
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run,mpLocalMapper);
    mpLocalMapper->mInitFr = 0; // seems to be ununsed
    if(settings_)
        mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    if(mpLocalMapper->mThFarPoints!=0)
    {
        Verbose::PrintMess("Discard points further than " +to_string(mpLocalMapper->mThFarPoints) + " m from current camera", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;

    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                    mpAtlas, mpKeyFrameDatabase, std::string(), mSensor, settings_);
    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpLocalMapper->SetTracker(mpTracker);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,std::string(),settings_);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
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

tuple<Sophus::SE3f, bool,bool, unsigned long int, vector<float>> System::TrackMonocular(const cv::cuda::HostMem &im_managed, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{

    ZoneNamedN(TrackMonocular, "TrackMonocular", true);  // NOLINT: Profiler

    {
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
        auto lock = scoped_mutex_lock( mMutexMode );
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                this_thread::sleep_for(chrono::microseconds(1000));
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
        auto lock = scoped_mutex_lock( mMutexReset );
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            Verbose::PrintMess("SYSTEM-> Reseting active map in monocular case", Verbose::VERBOSITY_NORMAL);
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    auto [Tcw,id, isKeyframe] = mpTracker->GrabImageMonocular(im_managed,timestamp,filename);
    
    auto isBAComplete = mpTracker->isBACompleteForMap();
    auto computedScales = mpTracker->getMapScales();

    return {Tcw,isBAComplete,isKeyframe,id, computedScales};
}

void System::ActivateLocalizationMode()
{
    auto lock = scoped_mutex_lock( mMutexMode );
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
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
    auto lock = scoped_mutex_lock( mMutexReset );
    mbReset = true;
}

void System::ResetActiveMap()
{
    auto lock = scoped_mutex_lock( mMutexReset );
    mbResetActiveMap = true;
}

void System::Shutdown()
{
    {
        auto lock = scoped_mutex_lock( mMutexReset );
        mbShutDown = true;
    }

    Verbose::PrintMess("Shutdown", Verbose::VERBOSITY_NORMAL);

    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();

}

bool System::isShutDown() {
    auto lock = scoped_mutex_lock( mMutexReset );
    return mbShutDown;
}

//This is not really correct -> remove in future use bool value from tracking
unsigned int System::GetLastKeyFrameId()
{
  auto lock = scoped_mutex_lock( mMutexState );
  return mpTracker->GetLastKeyFrameId();
}

cv::Mat System::DrawTrackedImage()
{
  auto lock = scoped_mutex_lock( mMutexState );
  return mpFrameDrawer->DrawFrame();
}


int System::GetTrackingState()
{
    auto lock = scoped_mutex_lock( mMutexState );
    return mpTracker->getTrackingState();
}

vector<MapPoint*> System::GetActiveReferenceMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    return pActiveMap->GetReferenceMapPoints();
}

std::shared_ptr<std::vector<KeyPoint>> System::GetTrackedKeyPointsUn()
{
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

bool System::isGeoreferenced() const {
    return mpTracker->isGeoreferenced();
}

bool System::isImuInitialized() const {
    return mpAtlas->isImuInitialized();
}

shared_ptr<mutex> System::getGlobalDataMutex(){
    return mpLocalMapper->getGlobalDataMutex();
}

void System::setGeoreference(bool is_georeferenced){
    mpTracker->setGeoreference(is_georeferenced);
}

vector<pair<long unsigned int,Sophus::SE3f>> System::getLatestOptimizedKFPoses() {
    return mpLocalMapper->getLatestOptimizedKFPoses();
}


} //namespace ORB_SLAM

