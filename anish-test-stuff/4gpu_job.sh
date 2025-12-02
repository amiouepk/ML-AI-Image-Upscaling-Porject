#!/bin/bash
#SBATCH -A am3138_g   # e.g., m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -n 4                     # Total number of tasks (1 node * 4 GPUs)
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32                    # CPU cores per task
#SBATCH --job-name=edsr-1node

# Load environment
module load pytorch

# Set master address for communication (required for DDP)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run the script
# -u allows python output to print immediately to logs
srun -u python train_ddp.py
