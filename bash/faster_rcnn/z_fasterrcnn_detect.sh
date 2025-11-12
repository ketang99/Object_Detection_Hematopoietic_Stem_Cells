#!/usr/bin/env bash
#SBATCH --gpus=V100:1
#SBATCH --mem=25000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/faster_rcnn/o_faster_rcnn_detect_z_e50_allclass_print.out

module load anaconda3
source activate myenv
python faster_rcnn/detect_faster_rcnn.py -a 0 -e 50 --model_folder faster_rcnn_50_Z_correctbigset_3classes_e50_o0_noroi15_earlystop1
