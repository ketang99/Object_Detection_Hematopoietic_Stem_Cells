#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=200000
#SBATCH --time=02:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_yolo_dataset_z_big_back.out

module load anaconda3
source activate myenv
python /home/kgupta/data/registration_testing/yolov5_new_back_create_patches.py -a 0 -s 1 -r 3 -d 2d_bigdataset_new

