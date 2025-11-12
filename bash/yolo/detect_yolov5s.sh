#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_val_yolo_best.out

module load anaconda3
source activate myenv
module load multigpu
python yolov5_z/val.py --img 640 --data yolo_dataset_Z/dataset.yaml --weights yolov5_z/runs/train/exp4/weights/best.pt
