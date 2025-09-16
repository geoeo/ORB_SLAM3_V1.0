#pragma once

namespace ros2_orbslam3
{
  /*!
  * @brief Class for WGS84 coordinates. 
  */

  class WGSPose
  {
    public:
      double latitude;
      double longitude;
      double altitude;
      double heading;
  };
}

