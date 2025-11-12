#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_detect_py_yolo_x_small_save_best_e300_bigset.out

module load anaconda3
source activate myenv
python detect_yolo_x.py
