#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --job-name=Infw/TTT
#SBATCH --output=slurm_output_%A.out

# Loading modules (CUDA and Anaconda are located in module 2024)
module load 2024
module load CUDA/12.6.0
module load Anaconda3/2024.06-1

# Activate conda environment
source activate dl2024

# Set working directory
cd "$TMPDIR" # $TMPDIR is set to /scratch-local/dlindberg

# Copy data to local scratch
if cp -r $HOME/first_dir/ARChitects/data "$TMPDIR"; then
    echo "data copied: $HOME/first_dir/ARChitects/data --> $TMPDIR/data"
else
    echo "Error: Failed to copy data from $HOME/first_dir/ARChitects/data to $TMPDIR"
    exit 1
fi

# Logging info
echo "Starting job at $(date)"

# Run the python script
srun python $HOME/first_dir/ARChitects/training_code/run_evaluation_Llama-rearc_with_ttt.py

# Copy output directory from scratch back to desired directory
cp -r output_evaluation_Llama-rearc_with_ttt $HOME/first_dir/ARChitects