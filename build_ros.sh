echo "Building ROS nodes"

source /opt/ros/humble/setup.bash
source /ros2_ws/install/local_setup.bash 
cd ros2_ws
rm -rf build install log
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=${1:-Release}
