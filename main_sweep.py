import os
import argparse
import torch
import torch.multiprocessing as mp
import hostlist
import wandb

from src.train_iterative import train_iterative
from src.args import parse_arguments

wandb.login(key=os.environ.get("WANDB_API_KEY", ""))

if __name__ == '__main__':
    args = parse_arguments()
    mp.set_start_method("spawn", force=True)

    NODE_ID = os.environ["SLURM_NODEID"]
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
    n_nodes = len(hostnames)
    print(NODE_ID, rank, local_rank, world_size, hostnames)

    # Handle SLURM_STEP_GPUS or fallback
    gpu_ids = os.getenv("SLURM_STEP_GPUS")
    if gpu_ids:
        gpu_ids = gpu_ids.split(",")
    else:  # Fallback: use torch to find visible GPUs if SLURM_STEP_GPUS is not set
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]

    wandb.init(config=args)

    train_iterative(args, device_id=gpu_ids)
