#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=25000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/faster_rcnn/o_faster_rcnn_dropout_overfit_e50_ef20.out

module load anaconda3
source activate myenv
python faster_rcnn/faster_rcnn_trainval_dropout.py -b 1 -o 1 --epochs_full 50 --epochs_frozen 20 --dropout 0.25

