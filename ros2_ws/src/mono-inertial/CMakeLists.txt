cmake_minimum_required(VERSION 3.8)
project(mono-inertial)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(OpenGL_GL_PREFERENCE "GLVND")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../../../Thirdparty/g2o/cmake_modules)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_common REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3.9 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)
find_package(ORB_SLAM3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(CUDACvManagedMemory REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_executable(mono_inertial_node src/mono_inertial_node.cpp)
target_link_libraries(mono_inertial_node 
  ${OpenCV_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDACvManagedMemory_LIBRARIES}
  ORB_SLAM3
  -lfmt
  )
ament_target_dependencies(mono_inertial_node rclcpp tf2_ros nav_msgs cv_bridge sensor_msgs ${TRACY_PACKAGE})

target_include_directories(mono_inertial_node PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
"$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${G2O_INCLUDE_DIR}
${CUDAToolkit_INCLUDE_DIRS}
${CUDACvManagedMemory_INCLUDE_DIRS}
)

# Install directories files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS mono_inertial_node DESTINATION lib/${PROJECT_NAME})

ament_package()
