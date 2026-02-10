#!/usr/bin/env bash

PROJECT_ROOT=$PWD
ROUNDS=${1:-0}

cd $PROJECT_ROOT/visiongl && ./run.sh $ROUNDS
cd $PROJECT_ROOT/sycl && ./run.sh $ROUNDS
cd $PROJECT_ROOT/cuda && ./run.sh $ROUNDS
cd $PROJECT_ROOT/matlab && ./run.sh $ROUNDS
