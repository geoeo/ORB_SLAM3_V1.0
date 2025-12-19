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

#include <vector>
#include <list>
#include <set>
#include <mutex>
#include <memory>

#include <KeyFrame.h>
#include <Frame.h>
#include <ORBVocabulary.h>
#include <Map.h>


namespace ORB_SLAM3
{

class KeyFrame;
class Frame;
class Map;


class KeyFrameDatabase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrameDatabase(){}
    KeyFrameDatabase(const std::shared_ptr<ORBVocabulary> voc);

    void add(std::shared_ptr<KeyFrame> pKF);

    void erase(std::shared_ptr<KeyFrame> pKF);

    void clear();
    void clearMap(std::shared_ptr<Map> pMap);

    // Loop Detection(DEPRECATED)
    std::vector<std::shared_ptr<KeyFrame>> DetectLoopCandidates(std::shared_ptr<KeyFrame> pKF, float minScore);

    // Loop and Merge Detection
    void DetectCandidates(std::shared_ptr<KeyFrame> pKF, float minScore,std::vector<std::shared_ptr<KeyFrame>>& vpLoopCand, std::vector<std::shared_ptr<KeyFrame>>& vpMergeCand);
    void DetectBestCandidates(std::shared_ptr<KeyFrame> pKF, std::vector<std::shared_ptr<KeyFrame>> &vpLoopCand, std::vector<std::shared_ptr<KeyFrame>> &vpMergeCand, int nMinWords);
    void DetectNBestCandidates(std::shared_ptr<KeyFrame> pKF, std::vector<std::shared_ptr<KeyFrame>> &vpLoopCand, std::vector<std::shared_ptr<KeyFrame>> &vpMergeCand, int nNumCandidates);

    // Relocalization
    std::vector<std::shared_ptr<KeyFrame>> DetectRelocalizationCandidates(std::shared_ptr<Frame> F, std::shared_ptr<Map> pMap);

protected:

   // Associated vocabulary
   const std::shared_ptr<ORBVocabulary> mpVoc;

   // Inverted file
   std::vector<std::list<std::shared_ptr<KeyFrame>> > mvInvertedFile;

   // For save relation without pointer, this is necessary for save/load function
   std::vector<std::list<long unsigned int> > mvBackupInvertedFileId;

   // Mutex
   std::mutex mMutex;

};

} //namespace ORB_SLAM
