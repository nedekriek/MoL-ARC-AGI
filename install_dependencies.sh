#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=00:30:00
#SBATCH --job-name=InstallEnvironment2
#SBATCH --output=slurm_output_%A.out

# Loading modules
module load 2024
module load Anaconda3/2024.06-1

cd $HOME/first_dir

conda env create -f environment.yml