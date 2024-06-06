cmake_minimum_required(VERSION 3.8)
project(mono-intertial)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

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
find_package(ORB_SLAM3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

add_executable(mono_intertial_node src/mono_intertial_node.cpp)
target_link_libraries(mono_intertial_node 
  ${OpenCV_LIBRARIES}
  ${Pangolin_LIBRARIES}
  ORB_SLAM3
  )
ament_target_dependencies(mono_intertial_node rclcpp tf2_ros nav_msgs cv_bridge sensor_msgs)

target_include_directories(mono_intertial_node PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
"$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
