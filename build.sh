echo "Configuring and building Thirdparty/DBoW2 ..."

export BUILD_TYPE=${1:-Release}
export G2O_FAST_MATH=${2:-OFF}

cd Thirdparty/DBoW2
cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_STANDARD=17
cmake --build build -j $(nproc --all) --target install

cd ../g2o

echo "Configuring and building Thirdparty/g2o ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_CXX_STANDARD=17 -DG2O_BUILD_APPS=OFF -DG2O_FAST_MATH=${G2O_FAST_MATH} -DBUILD_UNITTESTS=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_USE_OPENGL=OFF
cmake --build build -j $(nproc --all) --target install

cd ../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_SOPHUS_TESTS=OFF
cmake --build build -j $(nproc --all) --target install

cd ../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DTRACY_ENABLE=OFF
cmake --build build -j $(nproc --all) --target install