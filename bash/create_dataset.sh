#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH --time=02:00:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_create_dataset.out

source activate myenv
python /home/kgupta/data/registration_testing/red_dsb_traintestval.py 
