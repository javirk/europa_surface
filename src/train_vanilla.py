import os
import torch
import wandb
import random

from src.models.utils import (get_datasets, get_loss_fn, build_optimizer_scheduler, get_model,
                              get_semanticseg_transformations)
from src.models.eval import evaluate


def train(args, initial_epoch=0, wandb_step=0, fold_number=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = list(range(torch.cuda.device_count()))
    args.fold_number = fold_number
    transforms = get_semanticseg_transformations()
    dataset = get_datasets(args, transformations=transforms, shuffle_training=True)
    args.num_classes = dataset.train_dataset.num_classes
    args.ignore_index = dataset.train_dataset.ignore_index

    if 'fold_' in args.save:
        args.save = args.save.replace(f'fold_{fold_number - 1}', f'fold_{fold_number}')
    else:
        args.save = os.path.join(args.save, f'fold_{fold_number}')
    if args.wandb:
        wandb.config.update(args, allow_val_change=True)

    model = get_model(args, dataset.train_dataset)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.to(device)
    model.train()

    num_batches = len(dataset.train_loader)

    loss_fn = get_loss_fn(args)
    optimizer, scheduler = build_optimizer_scheduler(args, model, num_batches)

    evaluate(model, args, args.eval_datasets, {'epoch': initial_epoch, 'step': 0}, prompting_eval=['none'])

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        loss_sum = 0
        for i, data in enumerate(dataset.train_loader):
            optimizer.zero_grad()

            inp = data['image'].to(device)
            original_size = inp.shape[-2:]  # They should all have the same shape. We hope
            if args.prompting:
                if random.random() < 0.5:
                    low_res_target = data['mask_bb_downsampled'].to(device)
                    boxes = data['boxes'].to(device)
                    point = None
                    target = data['mask_bb'].to(device)
                else:
                    low_res_target = data['mask_point_downsampled'].to(device)
                    boxes = None
                    point = (data['point'].to(device), data['point_label'].to(device))
                    target = data['mask_point'].to(device)
            else:
                low_res_target = data['mask_downsampled'].to(device)
                target = data['mask'].to(device)
                boxes, point = None, None

            output = model(inp, original_size, boxes, point)

            loss = 0
            losses = {}
            for name, (w, fn) in loss_fn.items():
                if 'iou' in name.lower():
                    loss_item = fn(output, low_res_target)
                else:
                    loss_item = fn(output['masks'], target)
                losses[f'train/{name}'] = loss_item.item()
                loss += loss_item * w
            loss.backward()

            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            wandb_step += 1

        current_epoch = epoch + initial_epoch + 1

        if args.wandb:
            wandb.log({'train/loss': loss_sum / num_batches}, step=wandb_step)

        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='val',
                 prompting_eval=['none'])
        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='train',
                  prompting_eval=['none'])
        # Saving models
        if args.save is not None and (epoch + 1) % args.write_freq == 0:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            torch.save(model.module.state_dict(), model_path)

    # Saving models
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'fold_{fold_number}.pt')
        torch.save(model.module.state_dict(), model_path)

    return wandb_step
