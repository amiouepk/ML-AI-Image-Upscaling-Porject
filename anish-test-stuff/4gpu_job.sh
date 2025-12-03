#!/bin/bash
#SBATCH -A m4431_g           # Your allocation (added _g for GPU access)
#SBATCH -C gpu               # Use GPU nodes
#SBATCH -q regular           # Queue type
#SBATCH -t 06:00:00          # Time limit (6 hours)
#SBATCH -N 1                 # 1 Node
#SBATCH --ntasks-per-node=4  # 4 Tasks (1 per GPU)
#SBATCH --gpus-per-node=4    # 4 GPUs total
#SBATCH -c 32                # 32 CPU cores per task (Crucial for num_workers=30)
#SBATCH --gpu-bind=none      # Let PyTorch handle binding
#SBATCH --output=logs/train_%j.out  # Standard output log
#SBATCH --error=logs/train_%j.err   # Error log

# 1. Create logs directory so the script doesn't fail
mkdir -p logs

# 2. Load the PyTorch module
module load pytorch/2.1.0-cu12

# 3. Ensure Kaggle is installed (for the dataset download)
pip install kaggle --user

# 4. Optimization settings
# Force Python to print logs immediately (no buffering)
export PYTHONUNBUFFERED=1
# Prevent CPU thread contention
export OMP_NUM_THREADS=1 

# 5. Run the script
echo "------------------------------------------------"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "------------------------------------------------"

# srun launches 4 copies of your script (one for each GPU)
srun python train_ddp.py

echo "------------------------------------------------"
echo "Job finished at $(date)"
echo "------------------------------------------------"