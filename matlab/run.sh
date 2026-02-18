#!/usr/bin/env bash

PRELUDE="warning('off','all'); parallel.gpu.enableCUDAForwardCompatibility(true);"
TXT_FILENAME="benchmark.txt"
CSV_FILENAME="benchmark.csv"
LOG_FILENAME="benchmark.log"
OUTPUT_FOLDER_BASE="../results"
IMAGE_PATTERN="../assets/mitosis/mitosis-5d%04d.tif"
INDEX_0=0
INDEX_N=335
ROUNDS=${1:-0}

OUTPUT_FOLDER=$OUTPUT_FOLDER_BASE/matlab
TECH_NAME="MATLAB"
mkdir -p "$OUTPUT_FOLDER/1D" "$OUTPUT_FOLDER/2D" "$OUTPUT_FOLDER/3D" "$OUTPUT_FOLDER/4D" "$OUTPUT_FOLDER/5D"
echo $TECH_NAME > "$OUTPUT_FOLDER/$TXT_FILENAME"
echo "Running $TECH_NAME 1D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER/1D\", 22020096);"           > "$OUTPUT_FOLDER/1D/$CSV_FILENAME" 2> "$OUTPUT_FOLDER/1D/$LOG_FILENAME"
echo "Running $TECH_NAME 2D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER/2D\", 256, 86016);"         > "$OUTPUT_FOLDER/2D/$CSV_FILENAME" 2> "$OUTPUT_FOLDER/2D/$LOG_FILENAME"
echo "Running $TECH_NAME 3D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER/3D\", 256, 256, 336);"      > "$OUTPUT_FOLDER/3D/$CSV_FILENAME" 2> "$OUTPUT_FOLDER/3D/$LOG_FILENAME"
echo "Running $TECH_NAME 4D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER/4D\", 256, 256, 2, 168);"   > "$OUTPUT_FOLDER/4D/$CSV_FILENAME" 2> "$OUTPUT_FOLDER/4D/$LOG_FILENAME"
echo "Running $TECH_NAME 5D benchmark"
matlab -batch "$PRELUDE benchmark(\"$IMAGE_PATTERN\", $INDEX_0, $INDEX_N, $ROUNDS, \"$OUTPUT_FOLDER/5D\", 256, 256, 2, 24, 7);" > "$OUTPUT_FOLDER/5D/$CSV_FILENAME" 2> "$OUTPUT_FOLDER/5D/$LOG_FILENAME"
echo "Results and logs saved in $(realpath $OUTPUT_FOLDER)"
