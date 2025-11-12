#!/usr/bin/env bash
#SBATCH --gpus=V100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/yolo/o_z_latest0828_yolov5_bigset_r3_e300_b32.out

module load anaconda3
source activate myenv
module load multigpu
python -m torch.distributed.run --nproc_per_node 2 yolov5_z_small/train.py --img 192 --batch 32 --epochs 300 --data z_2d_bigdataset_new_Z_r3/dataset.yaml --weights yolov5n.pt
