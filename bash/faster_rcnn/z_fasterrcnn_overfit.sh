#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/faster_rcnn/o_faster_rcnn_z_overfit_ignoreback_e100_ef15.out

module load multigpu
module load anaconda3
source activate myenv
python faster_rcnn/faster_rcnn_trainval.py -b 1 -o 1 -e 100 -w 15 --stop_training 0 -a 0
~
