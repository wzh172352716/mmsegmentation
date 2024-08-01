#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --output=output_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load necessary modules
# module load anaconda/3-4.9.2 
module load cuda/12.2

conda init
# Activate conda environment
conda activate openmmlab

# Verify CUDA setup
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
