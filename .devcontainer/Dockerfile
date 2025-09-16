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
COPY .devcontainer/build_scripts/install_ros.sh /root
WORKDIR /root
RUN bash install_ros.sh

# Eigen3 is getting installed by some ROS package. Remove here so that we dont have runtime issues
RUN rm -rf /usr/include/eigen3 && rm -rf /usr/share/eigen3

WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2.tar.gz
RUN tar -xzvf cmake-3.25.2.tar.gz
RUN cd cmake-3.25.2 && ./bootstrap && make -j$(nproc) && make install
RUN rm cmake-3.25.2.tar.gz


WORKDIR /root
RUN apt update && sudo apt install -y libgtk2.0-dev
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip 
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.9.0.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN cd opencv-4.9.0 && mkdir -p build && cd build \
&& cmake \
    -DWITH_CUBLAS=ON \
    -DWITH_CUDA=ON \
    #-DCUDA_ARCH_BIN=8.7 \
    #-DCUDA_ARCH_PTX="sm_87" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_GENERATE_PKGCONFIG=OFF \
    -DBUILD_TESTS=OFF \
    -DCUDA_FAST_MATH=ON \
    #-DENABLE_NEON=ON \
    #-DBUILD_opencv_python2=OFF \
    #-DBUILD_opencv_python3=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.9.0/modules \
    -DCMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j $(nproc --all) && make install
RUN rm opencv_contrib.zip
RUN rm opencv.zip 

RUN git clone https://gitlab.com/libeigen/eigen.git \
    && cd eigen && git checkout 63a40ffb951a8f6b7662d14c6613c5c2994a9b61 \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build -j 6 --target install

# Deps for cv_image_transport - might be able to remove boost
RUN apt-get update && apt-get install -y \
    libturbojpeg0-dev \
    libogg-dev \
    libtheora-dev \
    libboost-all-dev \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/ORB_SLAM3_V1.0
COPY .devcontainer/build_scripts/install_ros_packages.sh /root
WORKDIR /root
RUN bash install_ros_packages.sh
    
WORKDIR /
# Sadly this is required as it is linked throught the project
RUN cd ~ && git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
RUN cd ~/Pangolin && git checkout d25ec00
RUN cd ~/Pangolin && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cd ~/Pangolin && cmake --build build -j $(nproc --all) --target install


# needed for g2o
RUN apt-get update && apt-get install -y libsuitesparse-dev 

# Tracy
RUN apt update && apt-get install -y libglfw3-dev libfreetype6 libcapstone-dev libtiff-dev libdbus-1-dev

# GIS conversions
RUN apt update && apt-get install -y libgdal-dev


WORKDIR /
RUN mkdir bags

