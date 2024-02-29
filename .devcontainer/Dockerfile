FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

# INSTALL SYSTEM DEPS + OpenCV Deps.
RUN apt-get update && apt-get install -y \
        build-essential \
        pkg-config \
        git \
        wget \
        curl \
        unzip \
        sudo \
        apt-utils \
        lsb-release \
        libglew-dev \
        gdb \
        libgtk2.0-dev \ 
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update &&\
    wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0-Linux-x86_64.sh &&\
    sh ./cmake.sh --prefix=/usr/local --skip-license

# For ROS with GUI support
ARG ROS_DISTRO=melodic

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends ros-melodic-desktop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN cd ~ && git clone -b 3.4.20 https://github.com/opencv/opencv.git \
    && cd opencv && mkdir build && cd build \
    && cmake -D WITH_CUDA=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_TESTS=OFF \
        -D CMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j $(nproc --all) && make install


# Deps for cv_image_transport
RUN apt-get update && apt-get install -y \
    python-catkin-tools \
    libturbojpeg0-dev \
    libogg-dev \
    libtheora-dev \
    #libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /
RUN mkdir -p /catkin_ws/src \
    && cd /catkin_ws/src \
    && git clone --branch melodic https://github.com/ros-perception/vision_opencv.git \
    && git clone --branch melodic-devel https://github.com/ros-perception/image_transport_plugins.git

WORKDIR /
RUN cd /catkin_ws && set -ex && . /opt/ros/${ROS_DISTRO}/setup.sh \
    && catkin config --extend /opt/ros/$ROS_DISTRO -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/local/share/OpenCV \
    && catkin build

WORKDIR /
# Sadly this is required as it is linked throught the project
RUN cd ~ && git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
RUN cd ~/Pangolin && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cd ~/Pangolin && cmake --build build -j $(nproc --all) --target install


WORKDIR /
RUN cd ~ && mkdir Eigen3 && cd Eigen3 \
    && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.bz2 \
    && tar xf eigen-3.3.9.tar.bz2 \
    && rm -rf eigen-3.3.9.tar.bz2 \
    && cd eigen-3.3.9 \
    && mkdir -p build && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. \
    && make -j $(nproc --all) \
    && make install

WORKDIR /
RUN mkdir bags

