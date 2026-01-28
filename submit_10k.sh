#!/bin/bash
#SBATCH --job-name=rumor_10k
#SBATCH --output=slurm_10k_%j.out
#SBATCH --error=slurm_10k_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# ============================================================================
# SLURM Job Script for 10,000 Iteration Rumor Simulation
# ============================================================================

echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="

# Load conda environment
source ~/.bashrc
conda activate mezo_llm

# Change to project directory
cd /home/chaymae.ed-dyb/project

# Run the 10k simulation
python run_10k.py

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
