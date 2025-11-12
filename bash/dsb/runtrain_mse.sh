#!/usr/bin/env bash
#SBATCH --gpus=A100:4
#SBATCH --mem=400000
#SBATCH --time=23:59:00
#SBATCH --output=/home/kgupta/data/registration_testing/models/bash/o_runtrain_test_7r_mseextract_lambda_sigmoid_completeset_scheduler_epoch100_sgd_2patch_e-2.out

module load anaconda3
source activate myenv
module load multigpu
TORCH_USE_CUDA_DSA=1 python dsb/7R_ND_HLF_MSE_extracted_full.py -e 100 -b 2 -o 1 -s 1 --lr 0.01

