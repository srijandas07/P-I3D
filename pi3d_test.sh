#!/usr/bin/env bash
#OAR -p gpu='YES' and gpucapability>'5.0'
#OAR -l /nodes=1/gpunum=4,walltime=72:00:00
#OAR --notify mail:akankshya.mishra@inria.fr
source ~/.bashrc

############# SET ENVIRONMENT #########################
module load cuda/10.0
module load cudnn/7.4-cuda-10.0
conda activate tensorflow-gpu1.13
#sudo mountimg right_hand.squashfs right_hand/
#sudo mountimg left_hand.squashfs left_hand/

############# TESTING SCRIPT ##########################
path=$1
python test.py $path

############ RESET ENVIRONMENT ########################
conda deactivate
