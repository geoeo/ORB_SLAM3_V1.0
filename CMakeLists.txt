cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

set(ORB_SLAM3_VERSION "1.0")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ORB_SLAM3_INC_INSTALL_DIR "/usr/local/include/${PROJECT_NAME}")
set(ORB_SLAM3_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
set(ORB_SLAM3_LIBRARY_INSTALL_DIR "/usr/local/lib")
set(ORB_SLAM3_ARCHIVE_INSTALL_DIR "/usr/local/lib")
set(ORB_SLAM3_FRAMEWORK_INSTALL_DIR "/usr/local/lib")

set(ORB_SLAM3_CMAKE_CONFIG_INSTALL_DIR "/usr/local/lib/cmake/${PROJECT_NAME}")
set(ORB_SLAM3_ADDITIONAL_FILES_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}")

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.4.20 REQUIRED EXACT)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 3.4.2 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.3.9 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
)

target_include_directories(${PROJECT_NAME} PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty> # for headers when building
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # for headers when building
     $<INSTALL_INTERFACE:${ORB_SLAM3_INC_INSTALL_DIR}> # for client in install mode
     ${EIGEN3_INCLUDE_DIR}
     ${Pangolin_INCLUDE_DIRS}
     ${DBoW2_INCLUDE_DIRS}
)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${Boost_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.a
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.a
-lcrypto
)


################################################################################
# Create Targets
################################################################################


# Library
install(TARGETS ${PROJECT_NAME} EXPORT ORB_SLAM3Targets
        RUNTIME       DESTINATION ${ORB_SLAM3_RUNTIME_INSTALL_DIR}
        LIBRARY       DESTINATION ${ORB_SLAM3_LIBRARY_INSTALL_DIR}
        ARCHIVE       DESTINATION ${ORB_SLAM3_ARCHIVE_INSTALL_DIR}
        FRAMEWORK     DESTINATION ${ORB_SLAM3_FRAMEWORK_INSTALL_DIR}
)

################################################################################
# Install Headers
################################################################################

file(GLOB HEAD 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

# Headers
install(
        FILES ${HEAD}
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
)

install(
        FILES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
)

install(
        DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CameraModels>
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
        PATTERN "*.hpp" # select header files
        PATTERN "lib" EXCLUDE
        PATTERN "build" EXCLUDE
)


install(
        DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2>
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
        PATTERN "*.hpp" # select header files
        PATTERN "lib" EXCLUDE
        PATTERN "build" EXCLUDE
)

install(
        DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/g2o>
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
        PATTERN "*.hpp" # select header files
        PATTERN "lib" EXCLUDE
        PATTERN "build" EXCLUDE
)

install(
        DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/Sophus>
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
        PATTERN "*.hpp" # select header files
        PATTERN "lib" EXCLUDE
        PATTERN "build" EXCLUDE
)

################################################################################
# Install All
################################################################################

# Create and install version file
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/ORB_SLAM3ConfigVersion.cmake")
set(config_file "${PROJECT_NAME}.cmake")

write_basic_package_version_file(${version_file}
        VERSION ${ORB_SLAM3_VERSION}
        COMPATIBILITY ExactVersion
        )

# install readme and license
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        DESTINATION ${ORB_SLAM3_ADDITIONAL_FILES_INSTALL_DIR}
        )

install(
        EXPORT ORB_SLAM3Targets
        DESTINATION ${ORB_SLAM3_CMAKE_CONFIG_INSTALL_DIR}
        FILE ORB_SLAM3Config.cmake
)


################################################################################
# Build Examples
################################################################################

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)


#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

