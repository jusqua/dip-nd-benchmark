#!/usr/bin/env bash

BUILD_FOLDER=./build
OUTPUT_FOLDER="./results"
RESULT_FOLDER="../results"
IMAGE_PATTERN="../assets/mitosis/mitosis-5d%04d.tif"
INDEX_0=0
INDEX_N=335
ROUNDS=${1:-0}

mkdir -p "$RESULT_FOLDER"

echo "Building VisionGL benchmark"
rm -rf $BUILD_FOLDER
cmake -G Ninja -S . -B $BUILD_FOLDER -D CMAKE_BUILD_TYPE=Release -D CMAKE_LINKER_TYPE=LLD -D CMAKE_CXX_COMPILER=clang++ > /dev/null 2>&1
cmake --build $BUILD_FOLDER > /dev/null 2>&1

# echo "Running VisionGL 1D benchmark"
# $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/visiongl-1d.csv" 2> "$RESULT_FOLDER/visiongl-1d.log"
echo "Running VisionGL 2D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/visiongl-2d.csv" 2> "$RESULT_FOLDER/visiongl-2d.log"
echo "Running VisionGL 3D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/visiongl-3d.csv" 2> "$RESULT_FOLDER/visiongl-3d.log"
echo "Running VisionGL 4D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/visiongl-4d.csv" 2> "$RESULT_FOLDER/visiongl-4d.log"
echo "Running VisionGL 5D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/visiongl-5d.csv" 2> "$RESULT_FOLDER/visiongl-5d.log"

echo "Results and logs saved in $RESULT_FOLDER"
