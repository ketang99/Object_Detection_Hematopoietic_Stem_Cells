#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000
#SBATCH --time=02:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_yolo_binary_z.out

module load anaconda3
source activate myenv
python /home/kgupta/data/registration_testing/0708_bintest2d.py
