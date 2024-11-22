#!/bin/bash
#( use ## for comments with SBATCH)
## DON'T USE SPACES AFTER COMMAS

# You must specify a valid email address!
#SBATCH --mail-user=javier.gamazo-tejero@unibe.ch
# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ws_00000

# Job name
#SBATCH --job-name="gal_sam_iterations"

# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long
##SBATCH --nodelist=gnode20

# Runtime and memory
#SBATCH --time=24:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=1G # it's memory PER CPU, NOT TOTAL RAM! maximum RAM is 246G in total
# total RAM is mem-per-cpu * cpus-per-task

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1

# on gpu partition
#SBATCH --gres=gpu:a100:1

# Set the current working directory.
# All relative paths used in the job script are relative to this directory
#SBATCH --output=logs/slurm-%A.out

# For array jobs
# Array job containing 100 tasks, run max 10 tasks at the same  time
##SBATCH --array=0-11%8

# Main Python code below this line
PYTHONPATH="./" python main.py --batch-size=24 --lr=5e-05 --wd=0.01 --epochs=50 --workers 8 \
  --data-location=/storage/workspaces/artorg_aimi/ws_00000/javier/datasets/europa/dataset_224x224/ \
  --eval-datasets=GalileoDataset --train-dataset=Galileo --loss-fn=DiceLoss,CrossEntropyLoss,IoUHeadLoss --loss-weights=1.,100.,1. \
  --wandb --exp-name=Galileo --pretrained-model=./ckpts/semseg.pt \
  --task training_iterative