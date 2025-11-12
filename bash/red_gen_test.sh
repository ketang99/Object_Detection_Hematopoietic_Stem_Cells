#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_red_gen_test_50.out

module load anaconda3
source activate myenv
python red_gen_test_arrays.py

