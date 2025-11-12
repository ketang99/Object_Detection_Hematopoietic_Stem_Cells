#!/usr/bin/env bash
#SBATCH --gpus=V100:2
#SBATCH --mem=200000
#SBATCH --time=04:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_detect_all_retina_x_12A.out

module load multigpu
module load anaconda3
source activate myenv
python detect_full_images.py -a 2 -m retina

