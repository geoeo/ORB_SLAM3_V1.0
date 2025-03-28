#!/bin/bash

echo "installing ros packages"
set -ex

mkdir -p /ros2_ws/src
apt update && apt install python3-numpy libboost-python-dev -y
cd /ros2_ws/src 

git clone https://github.com/ros-perception/vision_opencv.git -b humble
git clone https://github.com/ros-perception/image_common.git -b humble
git clone https://github.com/ros-perception/image_transport_plugins.git -b humble
cd /ros2_ws 

source /opt/ros/humble/setup.bash 
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source /ros2_ws/install/local_setup.bash 
colcon test
