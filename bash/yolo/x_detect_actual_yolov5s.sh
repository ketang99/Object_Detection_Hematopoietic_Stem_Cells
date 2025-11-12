#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_detect_yolo_x_small_save_best_e300_bigset.out

module load anaconda3
source activate myenv
python yolov5_x_small/detect.py --weights runs/detect/yolov5_x_small/runs/train/v8_exp3_e300_correct/weights/best.pt --source x_2d_bigdataset_new_X_r3/full_Test.txt --save-txt --save-csv --save-conf --imgsz 192
