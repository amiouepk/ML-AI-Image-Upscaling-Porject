#!/bin/bash
#SBATCH -A am3138_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 2                     # Request 2 Nodes
#SBATCH -n 8                     # Total tasks (2 nodes * 4 GPUs = 8)
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --job-name=edsr-2nodes

module load pytorch

# Setup Master Address (Getting the hostname of the first node)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

srun -u python train_ddp.py