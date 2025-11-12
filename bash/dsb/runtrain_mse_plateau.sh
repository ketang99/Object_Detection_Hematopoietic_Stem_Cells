#!/usr/bin/env bash
#SBATCH --gpus=A100:2
#SBATCH --mem=200000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_runtrain_test_mse_plateauscheduler_relu_epoch200_defanch_sgd_pneq_2patch_e-1.out

module load anaconda3
source activate myenv
module load multigpu
TORCH_USE_CUDA_DSA=1 python dsb/ND_HLF_MSE_plateau.py -b 2 --lr 0.1

