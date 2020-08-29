#!/bin/bash
#SBATCH --job-name=ML_cond
#SBATCH --mem-per-cpu=6000 
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1 
#SBATCH --array=0-100
#SBATCH --output=out.txt
#SBATCH --open-mode=append

source activate behnmf

bin=../scripts/morris_lecar_conductances.py

python $bin $SLURM_ARRAY_TASK_ID 

exit 0
