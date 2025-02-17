echo "Configuring and building Thirdparty/DBoW2 ..."

export BUILD_TYPE=${1:-Release}
shift

cd Thirdparty/DBoW2
cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} "$@"
make -C build

cd ../g2o

echo "Configuring and building Thirdparty/g2o ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/usr/local -DG2O_BUILD_APPS=OFF -DBUILD_WITH_MARCH_NATIVE=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_USE_OPENGL=OFF
cmake --build build -j $(nproc --all) --target install

cd ../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
cmake --build build -j 6 --target install

cd ../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DTRACY_ENABLE=ON "$@"
cmake --build build -j 6 --target install