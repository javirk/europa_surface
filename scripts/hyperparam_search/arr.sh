#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=javier.gamazo-tejero@unibe.ch
#SBATCH --mail-type=FAIL,END

# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long

# Runtime and memory
#SBATCH --time=3:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=4G

# on gpu partition
#SBATCH --gres=gpu:rtx3090:1

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1

#SBATCH --job-name=point_prompting_arr
#SBATCH --output=logs/slurm-%A.out
#SBATCH --array=1-30%2

# Replace <number_of_scripts> with the total number of script files you have.

#cd ../../

script_dir="./scripts/hyperparam_search"

script_files=($script_dir/point_*.sh)

current_script=${script_files[$SLURM_ARRAY_TASK_ID-1]}

# Check if the array task ID is within bounds
if [ "$SLURM_ARRAY_TASK_ID" -le ${#script_files[@]} ]; then
    echo "Running: $current_script"
    bash "$current_script"
else
    echo "Array task ID is out of bounds: $SLURM_ARRAY_TASK_ID"
fi
