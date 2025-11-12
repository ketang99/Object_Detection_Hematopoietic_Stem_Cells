#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=25000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/retinanet/o_retinanet_detect_z_complete_e100_earlystop1.out

module load anaconda3
source activate myenv
python retinanet/detect_retinanet.py -e 100 --stop_training 1 -o 0 -a 0  
