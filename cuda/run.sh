#!/usr/bin/env bash

BUILD_FOLDER=./build
OUTPUT_FOLDER="./results"
RESULT_FOLDER="../results"
IMAGE_PATTERN="../assets/mitosis/mitosis-5d%04d.tif"
INDEX_0=0
INDEX_N=335
ROUNDS=${1:-0}

mkdir -p "$RESULT_FOLDER"

echo "Building AdaptiveCPP PCUDA benchmark"
rm -rf $BUILD_FOLDER
cmake -G Ninja -S . -B $BUILD_FOLDER -D CMAKE_BUILD_TYPE=Release -D CMAKE_LINKER_TYPE=LLD -D CMAKE_CXX_COMPILER=acpp > "$RESULT_FOLDER/acpp-pcuda-build.log" 2>&1
cmake --build $BUILD_FOLDER >> "$RESULT_FOLDER/acpp-pcuda-build.log" 2>&1

# echo "Running AdaptiveCPP PCUDA L1 1D benchmark"
# ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/acpp-pcuda-l1-1d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l1-1d.log"
echo "Running AdaptiveCPP PCUDA L1 2D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/acpp-pcuda-l1-2d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l1-2d.log"
echo "Running AdaptiveCPP PCUDA L1 3D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/acpp-pcuda-l1-3d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l1-3d.log"
echo "Running AdaptiveCPP PCUDA L1 4D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/acpp-pcuda-l1-4d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l1-4d.log"
echo "Running AdaptiveCPP PCUDA L1 5D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/acpp-pcuda-l1-5d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l1-5d.log"

# echo "Running AdaptiveCPP PCUDA L2 1D benchmark"
# ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/acpp-pcuda-l2-1d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l2-1d.log"
echo "Running AdaptiveCPP PCUDA L2 2D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/acpp-pcuda-l2-2d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l2-2d.log"
echo "Running AdaptiveCPP PCUDA L2 3D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/acpp-pcuda-l2-3d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l2-3d.log"
echo "Running AdaptiveCPP PCUDA L2 4D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/acpp-pcuda-l2-4d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l2-4d.log"
echo "Running AdaptiveCPP PCUDA L2 5D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/acpp-pcuda-l2-5d.csv" 2> "$RESULT_FOLDER/acpp-pcuda-l2-5d.log"

echo "Building CUDA benchmark"
rm -rf $BUILD_FOLDER
cmake -G Ninja -S . -B $BUILD_FOLDER -D CMAKE_BUILD_TYPE=Release -D CMAKE_LINKER_TYPE=LLD -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_CUDA_COMPILER=nvcc > "$RESULT_FOLDER/cuda-build.log" 2>&1
cmake --build $BUILD_FOLDER >> "$RESULT_FOLDER/cuda-build.log" 2>&1

# echo "Running CUDA 1D benchmark"
# $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/cuda-1d.csv" 2> "$RESULT_FOLDER/cuda-1d.log"
echo "Running CUDA 2D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/cuda-2d.csv" 2> "$RESULT_FOLDER/cuda-2d.log"
echo "Running CUDA 3D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/cuda-3d.csv" 2> "$RESULT_FOLDER/cuda-3d.log"
echo "Running CUDA 4D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/cuda-4d.csv" 2> "$RESULT_FOLDER/cuda-4d.log"
echo "Running CUDA 5D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/cuda-5d.csv" 2> "$RESULT_FOLDER/cuda-5d.log"

echo "Results and logs saved in $RESULT_FOLDER"
