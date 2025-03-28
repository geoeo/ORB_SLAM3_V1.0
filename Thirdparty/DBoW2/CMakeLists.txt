cmake_minimum_required(VERSION 3.25)
project(DBoW2 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV QUIET REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

add_library(${CMAKE_PROJECT_NAME} SHARED)

target_include_directories(${CMAKE_PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${OpenCV_INCLUDE_DIRS}
)

set(HDRS_DBOW2
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/BowVector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/FORB.h 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/FClass.h       
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/FeatureVector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/ScoringObject.h   
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
${CMAKE_CURRENT_SOURCE_DIR}/src/DBoW2/BowVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DBoW2/FORB.cpp      
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DBoW2/FeatureVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DBoW2/ScoringObject.cpp
  )

set(HDRS_DUTILS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/DUtils/Random.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2/DUtils/Timestamp.h)
set(SRCS_DUTILS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DUtils/Random.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DUtils/Timestamp.cpp)

target_sources(${CMAKE_PROJECT_NAME}
  PRIVATE ${SRCS_DBOW2} ${SRCS_DUTILS}
  PUBLIC FILE_SET HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${HDRS_DUTILS} ${HDRS_DBOW2}
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
### INSTALLING ###

# install the headers that we specified with FILE_SET (header files for the user to include in their projects)
install(TARGETS ${CMAKE_PROJECT_NAME}
  EXPORT DBoW2Targets
  FILE_SET HEADERS
)


# provide some basic meta data about the library to generate the library's config file 
install(
  EXPORT DBoW2Targets
  FILE DBoW2Targets.cmake
  DESTINATION "lib/cmake/DBoW2"
)

# generate the library's config file
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${CMAKE_PROJECT_NAME}
    CACHE PATH "Location of header files" )
    
set(SYSCONFIG_LIBRARY_PATH ${CMAKE_INSTALL_LIBDIR}/lib${CMAKE_PROJECT_NAME}.so
    CACHE PATH "Location of configuration files" )

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DBoW2Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  INSTALL_DESTINATION "lib/cmake/DBoW2"
  PATH_VARS INCLUDE_INSTALL_DIR SYSCONFIG_LIBRARY_PATH
)

# install the config file
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION "lib/cmake/DBoW2"
)


