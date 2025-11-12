#!/usr/bin/env bash
#SBATCH --gpus=V100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_z_yolo8_cls_bigset_r3_e300_b32_pretrained.out

module load anaconda3
source activate myenv
module load multigpu
yolo detect train imgsz=192 batch=32 epochs=300 data=z_2d_bigdataset_new_Z_r3/dataset.yaml model=yolov8n-cls.pt name=yolov5_z_small/runs/train/v8_exp7_cls_e300_correct_pretrained pretrained=True plots=True save=True workers=2
