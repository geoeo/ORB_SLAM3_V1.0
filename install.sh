echo "Installing Thirdparty/DBoW2 ..."

export BUILD_TYPE=${1:-Release}
export G2O_FAST_MATH=${2:-OFF}

cd Thirdparty/DBoW2
cmake --install build

cd ../g2o

echo "Installing Thirdparty/g2o ..."

cmake --install build

cd ../Sophus

echo "Installing Thirdparty/Sophus ..."

cmake --install build

cd ../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Installing ORB_SLAM3 ..."

cmake --install build
