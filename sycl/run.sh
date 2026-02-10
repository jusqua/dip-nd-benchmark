#!/usr/bin/env bash

BUILD_FOLDER=./build
OUTPUT_FOLDER="./results"
RESULT_FOLDER="../results"
IMAGE_PATTERN="../assets/mitosis/mitosis-5d%04d.tif"
INDEX_0=0
INDEX_N=335
ROUNDS=${1:-0}

mkdir -p "$RESULT_FOLDER"

echo "Building AdaptiveCPP SYCL benchmark"
rm -rf $BUILD_FOLDER
cmake -G Ninja -S . -B $BUILD_FOLDER -D CMAKE_BUILD_TYPE=Release -D CMAKE_LINKER_TYPE=LLD -D CMAKE_CXX_COMPILER=acpp > "$RESULT_FOLDER/acpp-sycl-build.log" 2>&1
cmake --build $BUILD_FOLDER >> "$RESULT_FOLDER/acpp-sycl-build.log" 2>&1

# echo "Running AdaptiveCPP SYCL L1 1D benchmark"
# ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/acpp-l1-1d.csv" 2> "$RESULT_FOLDER/acpp-l1-1d.log"
echo "Running AdaptiveCPP SYCL L1 2D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/acpp-sycl-l1-2d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l1-2d.log"
echo "Running AdaptiveCPP SYCL L1 3D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/acpp-sycl-l1-3d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l1-3d.log"
echo "Running AdaptiveCPP SYCL L1 4D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/acpp-sycl-l1-4d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l1-4d.log"
echo "Running AdaptiveCPP SYCL L1 5D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=1 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/acpp-sycl-l1-5d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l1-5d.log"

# echo "Running AdaptiveCPP SYCL L2 1D benchmark"
# ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/acpp-sycl-l2-1d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l2-1d.log"
echo "Running AdaptiveCPP SYCL L2 2D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/acpp-sycl-l2-2d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l2-2d.log"
echo "Running AdaptiveCPP SYCL L2 3D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/acpp-sycl-l2-3d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l2-3d.log"
echo "Running AdaptiveCPP SYCL L2 4D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/acpp-sycl-l2-4d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l2-4d.log"
echo "Running AdaptiveCPP SYCL L2 5D benchmark"
ACPP_DEBUG_LEVEL=0 ACPP_ADAPTIVITY_LEVEL=2 $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/acpp-sycl-l2-5d.csv" 2> "$RESULT_FOLDER/acpp-sycl-l2-5d.log"

echo "Building Intel LLVM SYCL benchmark"
rm -rf $BUILD_FOLDER
cmake -G Ninja -S . -B $BUILD_FOLDER -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=icpx -D CMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvidia_gpu_sm_90" > "$RESULT_FOLDER/intel-sycl-build.log" 2>&1
cmake --build $BUILD_FOLDER >> "$RESULT_FOLDER/intel-sycl-build.log" 2>&1

# echo "Running Intel LLVM SYCL 1D benchmark"
# $BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 22020096       > "$RESULT_FOLDER/intel-sycl-1d.csv" 2> "$RESULT_FOLDER/intel-sycl-1d.log"
echo "Running Intel LLVM SYCL 2D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 86016      > "$RESULT_FOLDER/intel-sycl-2d.csv" 2> "$RESULT_FOLDER/intel-sycl-2d.log"
echo "Running Intel LLVM SYCL 3D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 336    > "$RESULT_FOLDER/intel-sycl-3d.csv" 2> "$RESULT_FOLDER/intel-sycl-3d.log"
echo "Running Intel LLVM SYCL 4D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 168  > "$RESULT_FOLDER/intel-sycl-4d.csv" 2> "$RESULT_FOLDER/intel-sycl-4d.log"
echo "Running Intel LLVM SYCL 5D benchmark"
$BUILD_FOLDER/benchmark $IMAGE_PATTERN $INDEX_0 $INDEX_N $ROUNDS $OUTPUT_FOLDER 256 256 2 24 7 > "$RESULT_FOLDER/intel-sycl-5d.csv" 2> "$RESULT_FOLDER/intel-sycl-5d.log"

echo "Results and logs saved in $(realpath $RESULT_FOLDER)"
