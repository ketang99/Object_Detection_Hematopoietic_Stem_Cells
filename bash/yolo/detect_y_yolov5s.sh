#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_detect_test_save_yolo_best.out

module load anaconda3
source activate myenv
python yolov5_y/detect.py --weights yolov5_y/runs/train/exp7/weights/best.pt --source yolo_dataset_Y/full_Test.txt --save-txt --save-csv --save-conf --save-crop

