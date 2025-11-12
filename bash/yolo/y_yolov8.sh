#!/usr/bin/env bash
#SBATCH --gpus=V100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_y_yolo8_bigset_r3_e300_b32.out

module load anaconda3
source activate myenv
module load multigpu
yolo detect train imgsz=192 batch=32 epochs=300 data=y_2d_bigdataset_new_Y_r3/dataset.yaml model=yolov8n.pt name=yolov5_y_small/runs/train/v8_exp3_e300_correct pretrained=False plots=True save=True workers=2

