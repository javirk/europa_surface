import os
import torch
import wandb

from src.models.utils import (get_datasets, get_loss_fn, build_optimizer_scheduler, get_model,
                              get_semanticseg_transformations)
from src.models.eval import evaluate
from src.datasets.base import DatasetBase


def bbox_iterative(args, initial_epoch=0, wandb_step=0, fold_number=0, device_id=None):
    if device_id is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_id}")
    args.device = device
    devices = list(range(torch.cuda.device_count()))
    args.fold_number = fold_number
    args.num_iterations = 5
    transforms = get_semanticseg_transformations()
    dataset = get_datasets(args, transformations=transforms, shuffle_training=True)
    args.num_classes = dataset.train_dataset.num_classes
    args.ignore_index = dataset.train_dataset.ignore_index

    if args.save is not None:
        if 'fold_' in args.save:
            args.save = args.save.replace(f'fold_{fold_number - 1}', f'fold_{fold_number}')
        else:
            args.save = os.path.join(args.save, f'fold_{fold_number}')
    if args.wandb:
        wandb.config.update(args, allow_val_change=True)

    model = get_model(args, dataset.train_dataset, train_encoder=True, train_prompt_encoder=True, train_decoder=True)
    # model = torch.nn.DataParallel(model, device_ids=devices)
    model.to(device)
    model.train()

    num_batches = len(dataset.train_loader)

    loss_fn = get_loss_fn(args)
    optimizer, scheduler = build_optimizer_scheduler(args, model, num_batches)

    evaluate(model, args, args.eval_datasets, {'epoch': initial_epoch, 'step': 0}, split='test', prompting_eval=['none'])
    evaluate(model, args, args.eval_datasets, {'epoch': initial_epoch, 'step': 0}, split='test', prompting_eval=['point'])
    evaluate(model, args, args.eval_datasets, {'epoch': initial_epoch, 'step': 0}, split='test', prompting_eval=['bb'])

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        loss_sum = 0
        for i, data in enumerate(dataset.train_loader):
            optimizer.zero_grad()

            inp = data['image'].to(device)
            original_size = inp.shape[-2:]  # They should all have the same shape. We hope

            none_target = data['mask'][:, 0].to(device)
            boxes = data['boxes'].to(device)
            current_boxes = boxes.clone()
            box_target = data['mask_point'][:, 0].to(device)  # Point target is actually the same as bbox target

            embeddings = model.encoder_step(inp)

            for i_iter in range(args.num_iterations):
                target = box_target
                # Iterations without prompts, to keep the model from overfitting to the prompts
                if i_iter > 0 and i_iter % (args.num_iterations // 2) == 0:
                    # This is an iteration without prompting
                    target = none_target
                    current_boxes = None

                output = model.from_embeddings(embeddings, original_size, current_boxes, points=None)

                loss = 0
                losses = {}
                for name, (w, fn) in loss_fn.items():
                    if 'iou' in name.lower():
                        loss_item = fn(output, target, use_low_res=False)
                    else:
                        loss_item = fn(output['masks'], target)
                        # loss_item = fn(output['low_res_logits'], low_res_target)

                    if f'train/{name}' not in losses:
                        losses[f'train/{name}'] = loss_item.item() / args.num_iterations
                    else:
                        losses[f'train/{name}'] += loss_item.item() / args.num_iterations
                    loss += loss_item * w

                loss /= args.num_iterations
                loss.backward(retain_graph=True)

                # Perturb the initial bounding box a little bit
                boxes = DatasetBase.perturb_bouding_box(boxes, original_size=original_size)
                current_boxes = boxes.clone()

            optimizer.step()
            scheduler.step()

            loss_sum += loss.item()
            wandb_step += 1

        current_epoch = epoch + initial_epoch + 1

        if args.wandb:
            wandb.log({'train/loss': loss_sum / num_batches, **losses}, step=wandb_step)

        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='test',
                 prompting_eval=['point'])
        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='train',
                 prompting_eval=['point'])
        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='test',
                 prompting_eval=['none'])
        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='train',
                 prompting_eval=['none'])
        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='test',
                 prompting_eval=['bb'])
        evaluate(model, args, args.eval_datasets, {'epoch': epoch, 'step': wandb_step}, split='train',
                 prompting_eval=['bb'])
        # Saving models
        if args.save is not None and (epoch + 1) % args.write_freq == 0:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            torch.save(model.state_dict(), model_path)

    # Saving models
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'fold_{fold_number}.pt')
        torch.save(model.state_dict(), model_path)

    return wandb_step
