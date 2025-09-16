#pragma once

#include <limits>

namespace ros2_orbslam3
{
/**
 * @brief EPSG 3857 Pose-Point class
 */

class EPSGPose
{
    // Class definition
  public:

    /** Null constructor. Makes a 2D, invalid point object. */
    EPSGPose()
        : easting(0.0),
          northing(0.0),
          altitude(std::numeric_limits<double>::quiet_NaN()),
          heading(std::numeric_limits<double>::quiet_NaN())
    {
    }

    /** Copy constructor. */
    EPSGPose(const EPSGPose &that) = default;

    /** Create a 3-D grid point. */
    EPSGPose(double _easting,
             double _northing,
             double _altitude,
             double _heading)
        : easting(_easting), northing(_northing), altitude(_altitude), heading(_heading)
    {
    }

    // data members
    double easting;           ///< easting within grid [meters]
    double northing;          ///< northing within grid [meters]
    double altitude;          ///< altitude above ellipsoid [meters] or NaN
    double heading;

}; // class EPSGPose

}

#pragma once
