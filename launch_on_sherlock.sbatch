#!/bin/bash
#SBATCH --job-name=estimation_driver
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=20
#SBATCH --output=estimation_driver_sherlock_bboptim.log
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL

# load modules
ml julia/1.2.0
# execute script
julia --machine-file <(srun hostname -s)  $GROUP_HOME/stb-model/code/estimation_driver_sherlock_bboptim.jl
