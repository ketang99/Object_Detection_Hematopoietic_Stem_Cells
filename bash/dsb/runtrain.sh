#!/usr/bin/env bash
#SBATCH --gpus=A100:3
#SBATCH --mem=300000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_runtrain_7r_completeset_defloss_sgd_red_dsb_epoch100_pneq_2patch_e-1.out

module load anaconda3
source activate myenv
module load multigpu 
TORCH_USE_CUDA_DSA=1 python dsb/7R_ND_HLF.py -b 32 -e 25 -d 1 -o 0 --lr 0.1 
