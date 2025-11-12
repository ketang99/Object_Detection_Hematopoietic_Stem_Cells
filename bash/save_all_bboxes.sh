#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000
#SBATCH --time=02:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_all_bboxes.out

module load anaconda3
source activate myenv
python /home/kgupta/data/registration_testing/bbox_save_all.py