#!/usr/bin/env bash
#SBATCH --gpus=V100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/faster_rcnn/o_faster_rcnn_z_complete__big_full_correctbigset_e50_ef15_earlystop.out

module load multigpu
module load anaconda3
source activate myenv
python faster_rcnn/faster_rcnn_trainval.py -b 32 -o 0 -e 50 -w 15 --stop_training 1 -a 0
