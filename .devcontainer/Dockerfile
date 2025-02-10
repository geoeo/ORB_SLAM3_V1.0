FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install locales -y
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN LANG=en_US.UTF-8

RUN apt install software-properties-common -y
RUN add-apt-repository universe

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
        g++ \
    && rm -rf /var/lib/apt/lists/*

# For ROS with GUI support
RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update && apt upgrade -y
RUN apt install ros-humble-ros-base -y
RUN apt install ros-dev-tools ros-humble-rosbag2-storage-mcap -y

WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2.tar.gz
RUN tar -xzvf cmake-3.25.2.tar.gz
RUN cd cmake-3.25.2 && ./bootstrap && make -j$(nproc) && make install
RUN rm cmake-3.25.2.tar.gz

WORKDIR /root
RUN apt update && sudo apt install -y libgtk2.0-dev
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip 
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.9.0.zip
RUN unzip opencv_contrib.zip
RUN unzip opencv.zip
RUN cd opencv-4.9.0 && mkdir -p build && cd build \
&& cmake -DWITH_CUDA=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_GENERATE_PKGCONFIG=OFF \
    -DBUILD_TESTS=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.9.0/modules \
    -DCMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j $(nproc --all) && make install
RUN rm opencv_contrib.zip
RUN rm opencv.zip 

# Deps for cv_image_transport
RUN apt-get update && apt-get install -y \
    libturbojpeg0-dev \
    libogg-dev \
    libtheora-dev \
    libeigen3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspaces/ORB_SLAM3_V1.0
COPY scripts/install_ros_packages.sh /root
WORKDIR /root
RUN bash install_ros_packages.sh
    
WORKDIR /
# Sadly this is required as it is linked throught the project
RUN cd ~ && git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
RUN cd ~/Pangolin && git checkout d25ec00
RUN cd ~/Pangolin && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cd ~/Pangolin && cmake --build build -j $(nproc --all) --target install

RUN cd ~ && mkdir Eigen3 && cd Eigen3 \
    && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2 \
    && tar xf eigen-3.4.0.tar.bz2 \
    && rm -rf eigen-3.4.0.tar.bz2 \
    && cd eigen-3.4.0 \
    && mkdir -p build && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. \
    && make -j $(nproc --all) \
    && make install

# needed for g2o
RUN apt-get update && apt-get install -y libsuitesparse-dev 

RUN git clone https://github.com/geoeo/CUDACvManagedMemory.git \
    && cd CUDACvManagedMemory \
    && cmake -S . -B build \
    && cmake --build build -j $(nproc --all) --target install

# Tracy
RUN apt update && apt-get install -y libglfw3-dev libfreetype6 libcapstone-dev libtiff-dev libdbus-1-dev

WORKDIR /
RUN mkdir bags

