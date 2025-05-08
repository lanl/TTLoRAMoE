#!/bin/bash

batch_sizes=(1 2 4 8 16 32 64 128)
dataset="qnli"

for batch in "${batch_sizes[@]}"
do
    echo "Running reconstruction test with batch size $batch"
    python Artifact_1.1/inference_comparison.py --batchsize $batch --dataset $dataset --test reconstruction --gpus 1 --workers 8

    echo "Running contraction test with batch size $batch"
    python Artifact_1.1/inference_comparison.py --batchsize $batch --dataset $dataset --test contraction --gpus 1 --workers 8
done
