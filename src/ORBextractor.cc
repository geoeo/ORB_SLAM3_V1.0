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

/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <execution>
#include <cuda_runtime.h>

#include "ORBextractor.h"
#include <tracy.hpp>

 	
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 25; // Seems to have be important for using the pattern

    float ORBextractor::IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
    {
        int m_01 = 0, m_10 = 0;

        const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return fastAtan2((float)m_01, (float)m_10);
    }



    ORBextractor::ORBextractor(int _nFeatures, int _nFastFeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST, int imageWidth, int imageHeight):
            nfeatures(_nFeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST),
            gpuFast(imageHeight, imageWidth ,_nFastFeatures), 
            gpuOrb(nfeatures),
            gpuAngle(nfeatures)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        for(int i=1; i<nlevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);
        mvBlurredImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }

        AllocatePyramid(imageWidth, imageHeight);
        ORB_SLAM3::cuda::angle::Angle::loadUMax(umax.data(), umax.size());
    }

    void ORBextractor::computeOrientation(const cv::Mat& image, std::vector<KeyPoint>& keypoints, const std::vector<int>& umax)
    {
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->angle = ORBextractor::IC_Angle(image, keypoint->pt, umax);
        }
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        //Associate points to childs
        for(size_t i=0;i<vKeys.size();i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if(kp.pt.x<n1.UR.x)
            {
                if(kp.pt.y<n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if(kp.pt.y<n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;

    }

    static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2){
        if(e1.first < e2.first){
            return true;
        }
        else if(e1.first > e2.first){
            return false;
        }
        else{
            if(e1.second->UL.x < e2.second->UL.x){
                return true;
            }
            else{
                return false;
            }
        }
    }

    vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const unsigned int fastKpCount, const short2 * location, const int* response, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        ZoneNamedN(DistributeOctTree, "DistributeOctTree", true);  // NOLINT: Profiler
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

        const float hX = static_cast<float>(maxX-minX)/nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for(int i=0; i<nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
            ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
            ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
            ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
            ni.vKeys.reserve(fastKpCount);

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        //Associate points to childs
        for(size_t i=0;i<fastKpCount;i++)
        {   
            const int kp_x = location[i].x;
            vpIniNodes[kp_x/hX]->vKeys.emplace_back(kp_x, location[i].y, -1, -1, static_cast<float>(response[i]));
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while(lit!=lNodes.end())
        {
            if(lit->vKeys.size()==1)
            {
                lit->bNoMore=true;
                lit++;
            }
            else if(lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        while(!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while(lit!=lNodes.end())
            {
                if(lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1,n2,n3,n4;
                    lit->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit=lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
            {
                bFinish = true;
            }
            else if(((int)lNodes.size()+nToExpand*3)>N)
            {

                while(!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                    for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                    {
                        ExtractorNode n1,n2,n3,n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                        // Add childs if they contain points
                        if(n1.vKeys.size()>0)
                        {
                            lNodes.push_front(n1);
                            if(n1.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n2.vKeys.size()>0)
                        {
                            lNodes.push_front(n2);
                            if(n2.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n3.vKeys.size()>0)
                        {
                            lNodes.push_front(n3);
                            if(n3.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n4.vKeys.size()>0)
                        {
                            lNodes.push_front(n4);
                            if(n4.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if((int)lNodes.size()>=N)
                            break;
                    }

                    if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                        bFinish = true;

                }
            }
        }

        // Retain the best point in each node
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        const int scaledPatchSize = static_cast<int>(PATCH_SIZE*mvScaleFactor[level]);
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint* pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;

            for(size_t k=1;k<vNodeKeys.size();k++)
            {
                if(vNodeKeys[k].response>maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }

            // Add border to coordinates and scale information
            pKP->octave=level;
            pKP->size = scaledPatchSize;

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }

    int ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
    {
        ZoneNamedN(ComputeKeyPointsOctTree, "ComputeKeyPointsOctTree", true);  // NOLINT: Profiler
        allKeypoints.resize(nlevels);
        const int BorderX = EDGE_THRESHOLD;
        const int BorderY = BorderX;

        for (int level = 0; level < nlevels; ++level)
        {
            const int width = mvImagePyramid[level]->getWidth();
            const int height = mvImagePyramid[level]->getHeight();
            unsigned int fastKpCount;

            ////////// Gpu Version //////////
            {
                ZoneNamedN(featCallGPU, "featCallGPU", true); 
                auto im_managed = mvImagePyramid[level]->getCvGpuMat(gpuFast.getStream());
                fastKpCount = gpuFast.detect(im_managed, iniThFAST, BorderX, BorderY);
                
                //Try again with lower threshold.
                if(fastKpCount == 0)
                    fastKpCount = gpuFast.detect(im_managed,minThFAST, BorderX, BorderY);
            }
            
            if(fastKpCount >0) {
                allKeypoints[level].reserve(nfeatures);
                allKeypoints[level] = DistributeOctTree(fastKpCount,gpuFast.getLoc(), gpuFast.getResp(), 0, width,
                                              0, height,mnFeaturesPerLevel[level], level);
            }

        }

        int allKeypointsCount = 0;
        // {
        //     ZoneNamedN(computeOrientationLoop, "computeOrientationLoop", true);  // NOLINT: Profiler
        //     // compute orientations
        //     for (int level = 0; level < nlevels; ++level){
        //         allKeypointsCount +=allKeypoints[level].size();
        //         ORBextractor::computeOrientation(mvImagePyramid[level]->getCvMat(), allKeypoints[level], umax);
        //     }
        // }

        {
            ZoneNamedN(computeOrientationLoop, "computeOrientationLoop", true);  // NOLINT: Profiler
            // compute orientations
            for (int level = 0; level < nlevels; ++level){
                const auto kpCount = allKeypoints[level].size();
                allKeypointsCount +=kpCount;
                if (kpCount > 0)
                    gpuAngle.launch_async(mvImagePyramid[level]->getCvGpuMat(gpuAngle.getStream()), allKeypoints[level].data(), allKeypoints[level].size(), HALF_PATCH_SIZE);
            }
        }
        return allKeypointsCount;
    }

    int ORBextractor::extractFeatures(const cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr &im_managed, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors)
    {
        ZoneNamedN(ApplyExtractor, "ApplyExtractor", true);  // NOLINT: Profiler

        //cv::Mat image = im_managed->getCvMat();
        //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
        //if(image.empty())
        //    return -1;

        assert(im_managed->getCvType() == CV_8UC1 );

        // Pre-compute the scale pyramid
        ComputePyramid(im_managed);

        vector < vector<KeyPoint> > allKeypoints;
        int nkeypoints = ComputeKeyPointsOctTree(allKeypoints);

        Mat descriptors;
        {
            ZoneNamedN(DescriptorAlloc, "DescriptorAlloc", true);
            if( nkeypoints == 0 )
                _descriptors.release();
            else
            {
                _descriptors.create(nkeypoints, 32, CV_8U);
                descriptors = _descriptors.getMat();
            }

            _keypoints = vector<cv::KeyPoint>(nkeypoints);
        }


        //Modified for speeding up stereo fisheye matching
        int offset = 0;
        int monoIndex = 0;
        {
            ZoneNamedN(DescriptorLoop, "DescriptorLoop", true);
            for (int level = 0; level < nlevels; ++level)
            {
                vector<KeyPoint>& keypointsLevel = allKeypoints[level];
                int nkeypointsLevel = (int)keypointsLevel.size();

                if(nkeypointsLevel==0)
                    continue;

                // preprocess the resized image
                {
                    ZoneNamedN(GaussianBlurCall, "GaussianBlurCall", true);
                    GaussianBlur(mvImagePyramid[level]->getCvMat(), mvBlurredImagePyramid[level]->getCvMat(), Size(7, 7), 1.2, 1.2, BORDER_REFLECT_101);
                }

                // Compute the descriptors - GPU
                {
                    ZoneNamedN(computeDescriptors, "computeDescriptors", true); 
                    cv::cuda::GpuMat gMat = mvBlurredImagePyramid[level]->getCvGpuMat(gpuOrb.getStream());
                    gpuOrb.launch_async(gMat, keypointsLevel.data(), keypointsLevel.size());
                    
                    Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
                    gpuOrb.join(desc);
    
                    offset += nkeypointsLevel;
    
                    float scale = mvScaleFactor[level]; 
                    for (size_t i = 0; i < keypointsLevel.size(); i++){
                        auto keypoint= keypointsLevel[i];
                        // Scale keypoint coordinates
                        keypoint.pt *= scale;
                        _keypoints.at(monoIndex) = keypoint;
                        desc.row(i).copyTo(descriptors.row(monoIndex));
                        monoIndex++;
                    }
                }
            }
        }

        return monoIndex;
    }

    void ORBextractor::AllocatePyramid(int width, int height)
    {
        ZoneNamedN(AllocatePyramid, "AllocatePyramid", true); 

        const auto size_in_bytes_level_0 = width*height;
        mvBlurredImagePyramid[0] = shared_ptr<cuda_cv_managed_memory::CUDAManagedMemory>(new cuda_cv_managed_memory::CUDAManagedMemory(size_in_bytes_level_0, height, width, CV_8UC1, width),cuda_cv_managed_memory::CUDAManagedMemoryDeleter());
        
        for (int level = 1; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            auto scaled_cols = scale*static_cast<float>(width);
            auto scaled_rows = scale*static_cast<float>(height);
            Size sz(cvRound(scaled_cols), cvRound(scaled_rows));

            const auto size_in_bytes = sz.height*sz.width;
            mvImagePyramid[level] = shared_ptr<cuda_cv_managed_memory::CUDAManagedMemory>(new cuda_cv_managed_memory::CUDAManagedMemory(size_in_bytes, sz.height, sz.width, CV_8UC1, sz.width),cuda_cv_managed_memory::CUDAManagedMemoryDeleter());
            mvBlurredImagePyramid[level] = shared_ptr<cuda_cv_managed_memory::CUDAManagedMemory>(new cuda_cv_managed_memory::CUDAManagedMemory(size_in_bytes, sz.height, sz.width, CV_8UC1, sz.width),cuda_cv_managed_memory::CUDAManagedMemoryDeleter());
        }
    }

    void ORBextractor::ComputePyramid(cuda_cv_managed_memory::CUDAManagedMemory::SharedPtr image_managed)
    {
        ZoneNamedN(ComputePyramid, "ComputePyramid", true);  // NOLINT: Profiler
        mvImagePyramid[0] = image_managed;

        for (int level = 1; level < nlevels; ++level)
        {
            // Compute the resized image
            // Use Orb Stream for now
            cv::cuda::Stream cvStream = gpuOrb.getCvStream();
            cv::cuda::GpuMat gpu_mat_prior_level = mvImagePyramid[level-1]->getCvGpuMat(gpuOrb.getStream());
            auto managed_image_level = mvImagePyramid[level];
            cv::cuda::GpuMat gpu_mat_level = managed_image_level->getCvGpuMat(gpuOrb.getStream());
            cv::Size sz = cv::Size(managed_image_level->getWidth(), managed_image_level->getHeight());

            cv::cuda::resize(gpu_mat_prior_level,gpu_mat_level , sz, 0, 0, cv::InterpolationFlags::INTER_LINEAR, cvStream);
            cvStream.waitForCompletion();
        }

    }

} //namespace ORB_SLAM
