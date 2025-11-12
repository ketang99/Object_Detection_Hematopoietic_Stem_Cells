#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=100000
#SBATCH --time=01:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_intensity_check.out

module load anaconda3
source activate myenv
python /home/kgupta/data/registration_testing/intensity_check.py