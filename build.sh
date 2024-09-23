echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=${1:-Release}
make -j6

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

cmake -S . -B build -DCMAKE_BUILD_TYPE=${1:-Release}\
 -DCMAKE_INSTALL_PREFIX=/usr/local -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_USE_OPENGL=OFF -DBUILD_WITH_MARCH_NATIVE=ON
cmake --build build -j $(nproc --all) --target install

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=${1:-Release}
make -j6

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=${1:-Release}
make -j6
make install
