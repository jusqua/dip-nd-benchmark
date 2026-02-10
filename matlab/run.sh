#!/usr/bin/env bash

PRELUDE="warning('off','all'); parallel.gpu.enableCUDAForwardCompatibility(true);"
OUTPUT_FOLDER="./results"
RESULT_FOLDER="../results"
IMAGE_PATTERN="../assets/mitosis/mitosis-5d%04d.tif"
INDEX_0=0
INDEX_N=335
ROUNDS=${1:-0}

mkdir -p "$RESULT_FOLDER"

# echo "Running MATLAB 1D benchmark"
# matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER\", 22020096);"           > "$RESULT_FOLDER/matlab-1d.csv" 2> "$RESULT_FOLDER/matlab-1d.log"
echo "Running MATLAB 2D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER\", 256, 86016);"         > "$RESULT_FOLDER/matlab-2d.csv" 2> "$RESULT_FOLDER/matlab-2d.log"
echo "Running MATLAB 3D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER\", 256, 256, 336);"      > "$RESULT_FOLDER/matlab-3d.csv" 2> "$RESULT_FOLDER/matlab-3d.log"
echo "Running MATLAB 4D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER\", 256, 256, 2, 168);"   > "$RESULT_FOLDER/matlab-4d.csv" 2> "$RESULT_FOLDER/matlab-4d.log"
echo "Running MATLAB 5D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER\", 256, 256, 2, 24, 7);" > "$RESULT_FOLDER/matlab-5d.csv" 2> "$RESULT_FOLDER/matlab-5d.log"

echo "Results and logs saved in $RESULT_FOLDER"
