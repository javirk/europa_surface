import os
import torch
import wandb
from datetime import datetime

from src.train_vanilla import train
# from src.test_segmentation import testing
from src.args import parse_arguments
from src.models.utils import write_results, make_name, project_name_composer
from src.test_segmentation import testing
from src.train_iterative import train_iterative
from src.generate_instance_segmentation import instance_seg
from src.bounding_box_prompting import bounding_box_prompt


def main(args):
    results = None
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    name = make_name(args, current_time)
    if args.save:
        args.save = os.path.join(args.save, args.exp_name, name)
        os.makedirs(args.save, exist_ok=True)
        # Save the args config to a file
        with open(os.path.join(args.save, 'config.txt'), 'w') as f:
            f.write(str(args))
    print(f"Using device: {args.device}")

    if args.task == 'testing':
        testing(args)
        # raise NotImplementedError("Testing not implemented yet.")
    elif args.task == 'instance_testing':
        instance_seg(args)
        return
    elif args.task == 'bounding_box_prompting':
        bounding_box_prompt(args)
        return


    # Training things
    if args.wandb:
        project_name = project_name_composer(args)
        wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
        wandb.init(
            # dir=os.path.join(args.save, 'wandb'),
            name=name,
            project=project_name,
            settings=wandb.Settings(start_method="spawn"),
            config=vars(args)
        )

    args.world_size = torch.cuda.device_count()

    if args.task == 'training':
        wandb_step = 0
        for fold_number in range(1):  # The dataset splits
            wandb_step = train(args, wandb_step=wandb_step, fold_number=fold_number)
    elif args.task == 'training_iterative':
        train_iterative(args, fold_number=0)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
