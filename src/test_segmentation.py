import os
import torch

import src.datasets as datasets
from src.models.utils import get_model, write_results
from src.models.eval import eval_single_dataset


def testing(args):
    args.wandb = False
    assert args.exp_name is not None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = list(range(torch.cuda.device_count()))

    ckpt = args.lora_ckpt if args.lora_ckpt is not None else args.testing_ckpt

    if os.path.isdir(ckpt):
        # Splits directory. Use all of them
        ckpts = [os.path.join(ckpt, x) for x in os.listdir(ckpt) if '.pt' in x or '.pth' in x]
    else:
        ckpts = [ckpt]

    for i, dataset_name in enumerate(args.eval_datasets):
        dataset_class = getattr(datasets, dataset_name)
        # Fold number is not important becase the test set is the same for all folds
        dataset = dataset_class(root=args.data_location, split='test', fold_number=0)
        args.num_classes = dataset.num_classes
        args.ignore_index = dataset.ignore_index

        for fold_ckpt in ckpts:
            fold_num = int(fold_ckpt.split('fold_')[-1].split('.pt')[0]) if 'fold' in fold_ckpt else fold_ckpt
            # model = sam_model_registry[args.model](checkpoint=args.pretrained_model, num_classes=dataset.num_classes,
            #                                        image_size=dataset.image_size)
            # if args.lora_ckpt is not None:
            #     model = LoRA_Sam(model, r=4)
            #     model.load_lora_parameters(fold_ckpt, device)
            args.lora_ckpt = fold_ckpt
            model = get_model(args, dataset)

            if args.testing_ckpt:
                model.load_state_dict(torch.load(fold_ckpt, map_location=device))

            model = torch.nn.DataParallel(model, device_ids=devices)
            model.to(device)
            model.eval()

            results_none, results_per_tile = eval_single_dataset(args, model, dataset, prompting='none',
                                                                 save_output_mask=True)
            # results_bb = eval_single_dataset(args, model, dataset, prompting='bb', embedding=False)
            # results_point = eval_single_dataset(args, model, dataset, prompting='point', embedding=False)

            write_results(args, [results_per_tile], f'{dataset_name}')

            results = {**results_none}

            with open(os.path.join(args.save, f'result.txt'), 'a') as f:
                for key, value in results.items():
                    f.write(f'{dataset_name},fold,{fold_num},{args.model},{key},{value}\n')
