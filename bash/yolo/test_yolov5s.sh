#!/usr/bin/env bash
#SBATCH --gpus=V100:2
#SBATCH --mem=100000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_yolo_test.out

module load anaconda3
source activate myenv
module load multigpu
python -m torch.distributed.run --nproc_per_node 2 yolov5_test/train.py --img 640 --batch 32 --epochs 200 --data test_yolo_dataset/dataset.yaml --weights yolov5s.pt

python yolov5_small_test_z/train.py --img 160 --batch 32 --epochs 3 --data test_yolo_dataset_small_Z/dataset.yaml --weights yolov5s.pt
