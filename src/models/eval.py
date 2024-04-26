import os.path

import matplotlib.pyplot as plt
import wandb
import torch
from tqdm import tqdm
from segmentation_models_pytorch import metrics
from torch.utils.data import Subset

import src.datasets as datasets
from src.models.utils import compute_class_weights
from src.datasets.dataset_utils import none_collate


def eval_single_dataset(args, model, dataset, prompting='point', save_output_mask=False):
    print(f"Evaluating on {dataset} with {prompting} prompting")
    os.makedirs(os.path.join(args.save, 'output_masks'), exist_ok=True)
    model.eval()
    device = args.device
    batch_size = args.batch_size
    max_label = args.num_classes - 1

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=none_collate
    )

    tp, fp, fn, tn = None, None, None, None
    boxes, point = None, None

    all_results = {
        'iou_macro': {},
        'iou_weight': {},
    }

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
            inp = data['image'].to(device)

            if prompting == 'point':
                target = data['mask_point'].to(device)
                point = (data['point'].to(device), data['point_label'].to(device))
            elif prompting == 'bb':
                target = data['mask_bb'].to(device)
                boxes = data['boxes'].to(device)
            else:
                target = data['mask'].to(device)
            original_size = target.shape[-2:]  # They should all have the same shape. We hope
            # fourier = data['fourier'].to(device) if args.fourier else None

            output = model(inp, original_size, boxes, point)
            output_mask = output['masks']
            output_mask = torch.argmax(output_mask, dim=1, keepdim=True)

            tp_b, fp_b, fn_b, tn_b = metrics.get_stats(output_mask, target,
                                                       mode='multiclass' if args.num_classes > 1 else 'binary',
                                                       num_classes=args.num_classes, ignore_index=args.ignore_index)
            if tp is None:
                # Declare them now
                tp, fp, fn, tn = [tp_b], [fp_b], [fn_b], [tn_b]
            else:
                # Append to the lists
                tp.append(tp_b)
                fp.append(fp_b)
                fn.append(fn_b)
                tn.append(tn_b)
            for j in range(len(data['name'])):
                name = data['name'][j]
                pixels_class = torch.bincount(target[j].flatten(), minlength=args.num_classes)
                class_weights = compute_class_weights(args, target, pixels_class)
                iou_im_macro = metrics.iou_score(tp_b[j][1:], fp_b[j][1:], fn_b[j][1:], tn_b[j][1:],
                                                                   reduction='macro')
                iou_im_weight = metrics.iou_score(tp_b[j], fp_b[j], fn_b[j], tn_b[j],
                                                                    reduction='weighted', class_weights=class_weights)
                all_results['iou_macro'][name] = iou_im_macro.item()
                all_results['iou_weight'][name] = iou_im_weight.item()
                if save_output_mask:
                    output_mask = output_mask[j].cpu().numpy()
                    output_mask = output_mask.squeeze()
                    save_path = os.path.join(args.save, 'output_masks', name + '.png')
                    plt.imsave(save_path, output_mask, cmap='gray')

    tp = torch.cat(tp, dim=0)
    fp = torch.cat(fp, dim=0)
    fn = torch.cat(fn, dim=0)
    tn = torch.cat(tn, dim=0)

    iou = metrics.iou_score(tp, fp, fn, tn, reduction='macro')
    metrics_dataset = {f'iou_{prompting}': iou}
    return metrics_dataset, all_results


def evaluate(model, args, eval_datasets, train_stats={}, split='val', prompting_eval=('none', 'bb', 'point'),
             multimask=False):
    eval_fn = eval_single_dataset if not multimask else eval_single_dataset_multimask
    if eval_datasets is None:
        return {}, {}
    for i, dataset_name in enumerate(eval_datasets):
        print('\nEvaluating on ', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        # dataset = dataset_class(root=args.emb_location, root_base=args.data_location, split=split, only_fluid=False)
        dataset = dataset_class(root=args.data_location, split=split)

        results_none, results_none_fine = {}, {}
        results_bb, results_bb_fine = {}, {}
        results_point, results_point_fine = {}, {}

        if 'none' in prompting_eval:
            results_none, results_none_fine = eval_fn(args, model, dataset, prompting='none')
        if 'bb' in prompting_eval:
            results_bb, results_bb_fine = eval_fn(args, model, dataset, prompting='bb')
        if 'point' in prompting_eval:
            results_point, results_point_fine = eval_fn(args, model, dataset, prompting='point')
        results = {**results_none, **results_point, **results_bb}
        results_fine = {'none': results_none_fine, 'bb': results_bb_fine, 'point': results_point_fine}

        if args.wandb:
            wandb.log({split + '/' + str(dataset) + '_' + k: v for k, v in results.items()}, step=train_stats['step'])
        else:
            print(results)

        if 'loss' in results:
            train_stats[dataset_name + " Loss"] = results['loss']

    return results, results_fine


def eval_single_dataset_multimask(args, model, dataset, prompting='point'):
    print(f"Evaluating on {dataset} with {prompting} prompting")
    model.eval()
    device = args.device
    batch_size = args.batch_size + args.transformations_sample
    num_masks = args.num_classes
    max_label = args.num_classes - 1

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=none_collate
    )

    tp, fp, fn, tn = None, None, None, None
    boxes, point = None, None

    all_results = {
        'iou_macro': {},
        'iou_weight': {},
    }

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
            inp = data['image'].to(device)

            if prompting == 'point':
                target = data['mask_point'].to(device)
                point = (data['point'].to(device), data['point_label'].to(device))
            elif prompting == 'bb':
                target = data['mask_bb'].to(device)
                boxes = data['boxes'].to(device)
            else:
                target = data['mask'].to(device)

            # target = target.squeeze(1)
            target = (target > 0).long()
            original_size = target.shape[-2:]  # They should all have the same shape. We hope

            output = model(inp, original_size, boxes, point)
            output_mask = output['masks']
            output_ious = output['iou_predictions']  # [B, N]
            best_iou_idx = torch.argmax(output_ious, dim=1)
            output_mask = output_mask[torch.arange(output_mask.shape[0]), best_iou_idx]
            output_mask = output_mask.sigmoid().unsqueeze(1)
            # output_mask = torch.argmax(output_mask, dim=1, keepdim=True)

            tp_b, fp_b, fn_b, tn_b = metrics.get_stats(output_mask > 0.5, target, mode='binary')
            if tp is None:
                # Declare them now
                tp, fp, fn, tn = [tp_b], [fp_b], [fn_b], [tn_b]
            else:
                # Append to the lists
                tp.append(tp_b)
                fp.append(fp_b)
                fn.append(fn_b)
                tn.append(tn_b)
            for j in range(len(data['name'])):
                name = data['name'][j]
                iou_im_macro = metrics.iou_score(tp_b[j], fp_b[j], fn_b[j], tn_b[j], reduction='macro')
                all_results['iou_macro'][name] = iou_im_macro.item()

    tp = torch.cat(tp, dim=0)
    fp = torch.cat(fp, dim=0)
    fn = torch.cat(fn, dim=0)
    tn = torch.cat(tn, dim=0)

    iou = metrics.iou_score(tp, fp, fn, tn, reduction='macro')
    metrics_dataset = {f'iou_{prompting}': iou}
    return metrics_dataset, all_results