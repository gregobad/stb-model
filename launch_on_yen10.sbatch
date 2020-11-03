#!/bin/bash
#SBATCH --job-name=estimation_driver
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=24
#SBATCH --output=estimation_driver_server_mcmc.log
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL

# load modules
ml julia
# execute script
julia --machine-file <(srun hostname -s)  /ifs/gsb/gjmartin/STBnews/stb-model-discrete/code/estimation_driver_server_mcmc.jl
