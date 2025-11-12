#!/usr/bin/env bash
#SBATCH --gpus=V100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_y_yolo_small.out

module load anaconda3
source activate myenv
module load multigpu
python -m torch.distributed.run --nproc_per_node 2 yolov5_y_small/train.py --img 640 --batch 32 --epochs 25 --data yolo_dataset_small_Y/dataset.yaml --weights yolov5s.pt

