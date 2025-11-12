#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_detect_yolo_z_small_save_best_e500_bigset.out

module load anaconda3
source activate myenv
python yolov5_z_small/detect.py --weights yolov5_z_small/runs/train/exp21/weights/best.pt --source yolo_bigdataset_new_Z_r3/full_Test.txt --save-txt --save-csv --save-conf --imgsz 192
