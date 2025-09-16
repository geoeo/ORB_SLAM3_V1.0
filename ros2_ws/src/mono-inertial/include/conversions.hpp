#pragma once

#include <gdal/ogr_spatialref.h>
#include <gdal/gdal_priv.h>

#include <wgs84.hpp>
#include <epsg3857.hpp>

namespace ros2_orbslam3
{

EPSGPose convertToEPSGFromWGS84(const WGSPose &wgs) {
  OGRSpatialReference ogr_epsg;
  ogr_epsg.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  ogr_epsg.importFromEPSG(3857);

  OGRSpatialReference ogr_wgs;
  ogr_wgs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  ogr_wgs.SetWellKnownGeogCS("WGS84");

  OGRCoordinateTransformation* coord_trans = OGRCreateCoordinateTransformation(&ogr_wgs, &ogr_epsg);

  double x = wgs.longitude;
  double y = wgs.latitude;
  bool result = coord_trans->Transform(1, &x, &y);

  // Underlying library is C malloc, have to clean up the resulting pointer to avoid a leak
  delete coord_trans;

  if (!result) {
    throw(std::runtime_error("Error converting utm coordinates to wgs84: Transformation failed"));
  }

  return EPSGPose{x, y, wgs.altitude, wgs.heading};

}

}