#!/usr/bin/env bash
#SBATCH --gpus=V100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_z_yolo.out

module load anaconda3
source activate myenv
module load multigpu
python -m torch.distributed.run --nproc_per_node 2 yolov5_z/train.py --img 640 --batch 32 --epochs 100 --data yolo_dataset_small_Z/dataset.yaml --weights yolov5s.pt

