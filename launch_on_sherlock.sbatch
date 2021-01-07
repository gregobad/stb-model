#!/bin/bash
#SBATCH --job-name=estimation_driver
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=50
#SBATCH --output=estimation_driver_server_mcmc.log
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL

# load modules
ml julia
# execute script
# julia --machine-file <(srun hostname -s)  $HOME/stb-model/code/estimation_driver_server_mcmc.jl
julia code/estimation_driver_server_mcmc.jl

# create standard output plots
ml R/4.0.2
ml hdf5
Rscript code/post_processing/standard_output_julia.R 

# combine with data plots to create standard output
ml system
ml texlive/2019
cd output/standard_output_graphs/
# latex output/standard_output_graphs/combine_standard_plots.tex
pdflatex combine_standard_plots.tex
cd ../../
git add output/MCMC_chain1_20days.csv
git add output/moments.csv
git add output/standard_output_graphs/combine_standard_plots.pdf
git add output/standard_output_graphs/*.png