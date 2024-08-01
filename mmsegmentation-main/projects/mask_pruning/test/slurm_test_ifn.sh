#!/usr/bin/env bash

##GENERAL -----
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=32000M
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --job-name=pedestrianRecall
#SBATCH --output=projects/mask_pruning/test/logs/%j_unittest.out

##DEBUG -----
#SBATCH --partition=debug
#SBATCH --time=00:20:00


module load comp/gcc/11.2.0
module load anaconda
source activate openmmlab

srun pytest -s projects/mask_pruning/test/test_masks.py