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

#include <FrameDrawer.h>
#include <Tracking.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>

using namespace std;

namespace ORB_SLAM3
{

FrameDrawer::FrameDrawer(shared_ptr<Atlas> pAtlas):both(false),mpAtlas(pAtlas),mCounter(0)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC1, cv::Scalar(0));
    mImPrev = cv::Mat(480,640,CV_8UC1, cv::Scalar(0));
}

cv::Mat FrameDrawer::DrawFrame(float imageScale)
{
    cv::Mat im;
    cv::Mat imPrev;
    shared_ptr<vector<KeyPoint>> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints - TODO: rework
    shared_ptr<vector<KeyPoint>> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state
    vector<float> vCurrentDepth;

    vector<shared_ptr<MapPoint>> vpLocalMap;
    vector<KeyPoint> vMatchesKeys;
    vector<shared_ptr<MapPoint>> vpMatchedMPs;
    vector<KeyPoint> vOutlierKeys;
    vector<shared_ptr<MapPoint>> vpOutlierMPs;
    vector<long unsigned int> currIds;

    map<long unsigned int, cv::Point2f> projectPoints;
    map<long unsigned int, KeyPoint> matchedInImage;
    map<long unsigned int, KeyPoint> matchedInImagePrev;

    cv::Scalar standardColor(0,255,0);
    cv::Scalar odometryColor(255,0,0);
    cv::Scalar outlierColor(0,0,255);
    cv::Scalar noLandmarkColor(0,0,0);

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);
        mImPrev.copyTo(imPrev);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vTracks = mvTracks;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;

            vpLocalMap = mvpLocalMap;
            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;
            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;
            projectPoints = mmProjectPoints;
            matchedInImage = mmMatchedInImage;
            matchedInImagePrev = mmMatchedInImagePrev;
            currIds = mCurrIds;

        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    }
    
    // if(imageScale != 1.f)
    // {
    //     int imWidth = im.cols / imageScale;
    //     int imHeight = im.rows / imageScale;
    //     std::cout << "resize" << std::endl;
    //     cv::resize(im, im, cv::Size(imWidth, imHeight));
    // }

        // if(im.channels()<3) {
        //     std::cout << "cvt" << std::endl;
        cvtColor(im,im,cv::COLOR_GRAY2BGR); // Keypoints are drawn on a color image
        // }
        cvtColor(imPrev,imPrev,cv::COLOR_GRAY2BGR);


    // //Draw
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys->operator[](i).pt / imageScale;
                    pt2 = vCurrentKeys->operator[](vMatches[i]).pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys->operator[](i).pt;
                    pt2 = vCurrentKeys->operator[](vMatches[i]).pt;
                }
                cv::line(im,pt1,pt2,standardColor);
            }
        }
        for(vector<pair<cv::Point2f, cv::Point2f> >::iterator it=vTracks.begin(); it!=vTracks.end(); it++)
        {
            cv::Point2f pt1,pt2;
            if(imageScale != 1.f)
            {
                pt1 = (*it).first / imageScale;
                pt2 = (*it).second / imageScale;
            }
            else
            {
                pt1 = (*it).first;
                pt2 = (*it).second;
            }
            cv::line(im,pt1,pt2, standardColor,5);
        }

    }
    else if(state==Tracking::OK) //TRACKING
    {


        cv::Mat imCombined = cv::Mat::zeros(im.rows, im.cols + imPrev.cols, im.type());
        imPrev.copyTo(imCombined.rowRange(0, imPrev.rows).colRange(0, imPrev.cols));
        im.copyTo(imCombined.rowRange(0, im.rows).colRange(imPrev.cols, imPrev.cols + im.cols));

        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys->size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2,pt1Shifted, pt2Shifted;
                cv::Point2f point, pointShifted, pointPrev;
                const auto currId = currIds[i];
                const auto PrevPointFound = matchedInImagePrev.find(currId) != matchedInImagePrev.end();


                if(imageScale != 1.f)
                {
                    const auto p = vCurrentKeys->operator[](i).pt;
                    point = p / imageScale;
                    float px = p.x / imageScale;
                    float py = p.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;

                    pointShifted.x=point.x+imPrev.cols;
                    pointShifted.y=point.y;

                    pt1Shifted.x=pt1.x+imPrev.cols;
                    pt1Shifted.y=pt1.y;
                    pt2Shifted.x=pt2.x+imPrev.cols;
                    pt2Shifted.y=pt2.y;

                    if(PrevPointFound)
                    {
                        const auto pPrev = matchedInImagePrev[currId].pt;
                        pointPrev = pPrev / imageScale;
                    }
                }
                else
                {
                    const auto p = vCurrentKeys->operator[](i).pt;
                    point = p;
                    pt1.x=p.x-r;
                    pt1.y=p.y-r;
                    pt2.x=p.x+r;
                    pt2.y=p.y+r;

                    pointShifted.x=point.x+imPrev.cols;
                    pointShifted.y=point.y;

                    pt1Shifted.x=pt1.x+imPrev.cols;
                    pt1Shifted.y=pt1.y;
                    pt2Shifted.x=pt2.x+imPrev.cols;
                    pt2Shifted.y=pt2.y;

                    if(PrevPointFound)
                    {
                        const auto pPrev = matchedInImagePrev[currId].pt;
                        pointPrev = pPrev;
                    }

                }

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,standardColor);
                    cv::circle(im,point,2,standardColor,-1);
                    mnTracked++;

                    cv::rectangle(imCombined,pt1Shifted,pt2Shifted,standardColor);
                    cv::circle(imCombined,pointShifted,2,standardColor,-1);
                    if(PrevPointFound)
                        cv::line(imCombined, pointPrev, pointShifted, standardColor);

                    cv::imwrite("combined_frame_" + to_string(mCounter) + ".png", imCombined);

                }
                else if(vbVO[i]) // This is match to a "visual odometry" MapPoint created in the last frame - 0 Obs
                {
                    cv::rectangle(im,pt1,pt2,odometryColor);
                    cv::circle(im,point,2,odometryColor,-1);
                    mnTrackedVO++;
                }
                else{ // Feature without a landmark - MapPoint
                    cv::rectangle(im,pt1,pt2,noLandmarkColor);
                    cv::circle(im,point,2,noLandmarkColor,-1);
                }
            }
            
        }





        // Outlier coloring

        // for(int i=0;i<n;i++)
        // {
        //     cv::Point2f pt1,pt2;
        //     cv::Point2f point;
        //     if(imageScale != 1.f)
        //     {
        //         const auto p = vOutlierKeys.operator[](i).pt;
        //         point = p / imageScale;
        //         float px = p.x / imageScale;
        //         float py = p.y / imageScale;
        //         pt1.x=px-r;
        //         pt1.y=py-r;
        //         pt2.x=px+r;
        //         pt2.y=py+r;
        //     }
        //     else
        //     {
        //         const auto p = vOutlierKeys.operator[](i).pt;
        //         point = p;
        //         pt1.x=p.x-r;
        //         pt1.y=p.y-r;
        //         pt2.x=p.x+r;
        //         pt2.y=p.y+r;
        //     }
        //     cv::rectangle(im,pt1,pt2,outlierColor);
        //     cv::circle(im,point,2,outlierColor,-1);
     
        // }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(shared_ptr<Tracking> pTracker)
{
    unique_lock<mutex> lock(mMutex);
    //Variables for the new visualization
    mImPrev = mIm.clone();
    mmMatchedInImagePrev = mmMatchedInImage;

    pTracker->mImGrayViewer.copyTo(mIm);
    auto currentFrame = pTracker->mCurrentFrame;

    mvCurrentKeys=currentFrame->mvKeysUn;
    mmProjectPoints = currentFrame->mmProjectPoints;
    mvpLocalMap = pTracker->GetLocalMapMPS();
    mbOnlyTracking = pTracker->mbOnlyTracking;
    mvIniKeys=pTracker->mInitialFrame->mvKeysUn;
    mvCurrentTrackedMapPoints = currentFrame->mvpMapPoints;
    const auto mvCurrentOutliers = currentFrame->mvbOutlier;
    const auto lastProcessedState = pTracker->getTrackingState();

    N = mvCurrentKeys->size();

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);


    mmMatchedInImage.clear();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);
    mCurrIds.clear();
    mCurrIds.resize(N, -1);

    if(lastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            auto pMP = mvCurrentTrackedMapPoints[i];
            if(pMP)
            {
                if(!mvCurrentOutliers[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    mmMatchedInImage[pMP->mnId] = mvCurrentKeys->operator[](i);
                    mCurrIds[i] = pMP->mnId;
                }
                else
                {
                    mvpOutlierMPs.push_back(pMP);
                    mvOutlierKeys.push_back(mvCurrentKeys->operator[](i));
                }
            }
        }

    }
    mState=static_cast<int>(lastProcessedState);
    mCounter++;
}

} //namespace ORB_SLAM
