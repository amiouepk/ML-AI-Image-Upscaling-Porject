#!/bin/bash
#SBATCH -A m4431_g           # Your Account ID
#SBATCH -C gpu               # Use GPU nodes
#SBATCH -q regular           # 'regular' queue (allow up to 12 hours)
#SBATCH -t 04:00:00          # Set max time to 4 hours (adjust if needed)
#SBATCH -N 1                 # 1 Node
#SBATCH --ntasks-per-node=4  # 4 tasks (processes)
#SBATCH --gpus-per-node=4    # 4 GPUs
#SBATCH -c 32                # 32 CPU cores per task (Helps data loading!)
#SBATCH --gpu-bind=none      # Let PyTorch handle binding
#SBATCH --output=logs/training_%j.out  # Save logs to a folder
#SBATCH --error=logs/training_%j.err

# 1. Create logs folder so the script doesn't complain
mkdir -p logs

# 2. Load Environment
module load pytorch

# 3. Install Kaggle if missing (just in case)
pip install kaggle --user

# 4. Critical Performance Flag (See logs instantly)
export PYTHONUNBUFFERED=1

# 5. Run the script
# We use 'srun' so it knows to launch 4 copies (1 per GPU)
echo "Starting training at $(date)"
srun python train_ddp.py
echo "Training finished at $(date)"