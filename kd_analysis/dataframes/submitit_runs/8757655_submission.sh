#!/bin/bash

# Parameters
#SBATCH --array=0-23%24
#SBATCH --error=/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/pruneshift_jupyter/kd_analysis/dataframes/submitit_runs/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/pruneshift_jupyter/kd_analysis/dataframes/submitit_runs/%A_%a_0_log.out
#SBATCH --partition=lmbdlc_gpu-rtx2080
#SBATCH --signal=TERM@90
#SBATCH --time=100
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/pruneshift_jupyter/kd_analysis/dataframes/submitit_runs/%A_%a_%t_log.out --error /work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/pruneshift_jupyter/kd_analysis/dataframes/submitit_runs/%A_%a_%t_log.err --unbuffered /work/dlclarge2/agnihotr-shashank-pruneshift/debug/debug/bin/python -u -m submitit.core._submit /work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/pruneshift_jupyter/kd_analysis/dataframes/submitit_runs
