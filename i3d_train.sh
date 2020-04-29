#!/usr/bin/env bash
source ~/.bashrc

############# SET ENVIRONMENT #########################
conda activate tensorflow-gpu1.13

############# TESTING SCRIPT ##########################
mkdir -p logs/
mkdir -p weights/
dataset=$1
protocol=$2
part=$3
num_classes=$4
batch_size=$5
epochs=$6

python i3d_train.py $dataset $protocol $part $num_classes $batch_size $epochs

############ RESET ENVIRONMENT ########################
conda deactivate
