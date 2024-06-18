echo "Removing Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
rm -rf build
rm -rf lib

cd ../g2o


echo "Removing Thirdparty/g2o ..."

rm -rf build
rm -rf lib

cd ../Sophus

echo "Removing Thirdparty/Sophus ..."

rm -rf build

cd ../../

echo "Removing ORB_SLAM3 ..."

rm -rf build
