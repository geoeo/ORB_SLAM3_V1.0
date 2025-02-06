cmake_minimum_required(VERSION 3.25)
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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) #  avoid rpath stripping at installations

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/cmake_modules)


if(TRACY_ENABLE)
    message("Tracy Enabled...")
    LIST(APPEND CMAKE_INSTALL_RPATH  ${CMAKE_SOURCE_DIR}/../ros2_ws/install/tracy_profiler/lib)
    set(TRACY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ros2_ws/install/tracy_profiler/tracy/public")
    set(TRACY_LIB "${CMAKE_SOURCE_DIR}/ros2_ws/install/tracy_profiler/lib/libtracy_profiler.so")
    add_compile_definitions(TRACY_ENABLE)
    message(${TRACY_INCLUDE_DIR})
    message(${TRACY_LIB})
else()
    set(TRACY_INCLUDE_DIR "")
    set(TRACY_LIB "")
endif()
LIST(APPEND CMAKE_INSTALL_RPATH  /workspaces/ORB_SLAM3_V1.0/ros2_ws/install/tracy_profiler/lib)

set(OpenGL_GL_PREFERENCE "GLVND")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
message(FATAL_ERROR "OpenCV.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})


find_package(Eigen3 3.3.9 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(CUDACvManagedMemory REQUIRED)

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


message("---------------: ${BUILD_INTERFACE}")

include_directories(${TRACY_INCLUDE_DIR})

target_include_directories(${PROJECT_NAME} PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty> # for headers when building
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # for headers when building
     $<INSTALL_INTERFACE:${ORB_SLAM3_INC_INSTALL_DIR}> # for client in install mode
     ${EIGEN3_INCLUDE_DIR}
     ${Pangolin_INCLUDE_DIRS}
     ${DBoW2_INCLUDE_DIRS}
     ${G2O_INCLUDE_DIR}
     ${CUDACvManagedMemory_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${Boost_LIBRARIES}
${G2O_CORE_LIBRARY}
${G2O_STUFF_LIBRARY}
${G2O_SOLVER_CHOLMOD}
${G2O_SOLVER_CSPARSE}
${G2O_SOLVER_CSPARSE_EXTENSION}
${G2O_SOLVER_DENSE}
${G2O_SOLVER_PCG}
${G2O_SOLVER_SLAM2D_LINEAR}
${G2O_SOLVER_STRUCTURE_ONLY}
${G2O_SOLVER_EIGEN}
${G2O_TYPES_DATA}
${G2O_TYPES_ICP}
${G2O_TYPES_SBA}
${G2O_TYPES_SCLAM2D}
${G2O_TYPES_SIM3}
${G2O_TYPES_SLAM2D}
${G2O_TYPES_SLAM3D}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.a
${TRACY_LIB}
${CUDACvManagedMemory_LIBRARIES}
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
        DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/Sophus>
        DESTINATION ${ORB_SLAM3_INC_INSTALL_DIR}
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
        PATTERN "*.hpp" # select header files
        PATTERN "lib" EXCLUDE
        PATTERN "build" EXCLUDE
)

install(
        DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ros2_ws/install/tracy_profiler/tracy/public/tracy>
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


