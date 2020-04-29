#!/usr/bin/env bash
source ~/.bashrc

############# SET ENVIRONMENT #########################
conda activate tensorflow-gpu1.13

############# TESTING SCRIPT ##########################
split=$1
python lstm_train_attention.py $split

############ RESET ENVIRONMENT ########################
conda deactivate
