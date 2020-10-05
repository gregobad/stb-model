#!/bin/bash
#SBATCH --job-name=estimation_driver
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus_per_task=24
#SBATCH --output=estimation_driver_sherlock_bboptim.log

# load modules
ml julia/1.2.0
# execute script
julia --machine-file <(srun hostname -s)  ./estimation_driver_sherlock_bboptim.jl
