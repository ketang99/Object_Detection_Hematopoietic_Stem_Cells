#!/usr/bin/env bash
#SBATCH --gpus=V100:3
#SBATCH --mem=300000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/retinanet/o_retinanet_x_all_classes_complete_bigset_e100_earlystop.out

module load anaconda3
source activate myenv
module load multigpu
python retinanet/retinanet_trainval.py -b 16 -o 0 -e 100 --stop_training 1 -f 0 -a 2

