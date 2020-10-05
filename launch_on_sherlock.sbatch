#!/bin/bash
#SBATCH --job-name=estimation_driver
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --output=estimation_driver.log
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -i|--input)
    INPUT="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters
echo ${INPUT}
# load modules
ml julia/1.2.0
# execute script
julia --machine-file <(srun hostname -s)  ./estimation_driver_sherlock_bboptim.jl