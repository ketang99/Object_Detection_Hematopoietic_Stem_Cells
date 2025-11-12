#!/usr/bin/env bash
# SBATCH --cpus-per-task=1
# SBATCH --mem=75000
# SBATCH --time=23:59:00
# SBATCH --output=/home/kgupta/data/registration_testing/models/bash/faster_rcnn/o_faster_rcnn_detect_x_e50_allclass_print.out

module load anaconda3
source activate myenv
python faster_rcnn/detect_faster_rcnn.py -a 2 -e 50 --model_folder faster_rcnn_50_X_correctbigset_3classes_e50_o0_noroi15_earlystop1
