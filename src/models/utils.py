import os
import math
import wandb
import torch
import random
import pandas as pd
from einops import rearrange
from typing import Callable, Dict
import segmentation_models_pytorch as smp
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision.transforms import v2
from segmentation_models_pytorch import metrics
from segment_anything import sam_model_registry

import src.datasets as datasets
from src.models import LoRA_Sam


def get_testtime_transformations():
    return v2.Compose([
        v2.RandomApply([
            v2.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
            v2.GaussianBlur(kernel_size=(5, 5)),
            v2.RandomAdjustSharpness(sharpness_factor=2)
        ], p=0.8)
    ])
    # return v2.Identity()


def get_semanticseg_transformations():
    return v2.Compose([
        v2.RandomApply([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=16),
            v2.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0),
            v2.GaussianBlur(kernel_size=(5, 5)),
            v2.RandomAdjustSharpness(sharpness_factor=2),
        ], p=0.8)
    ])


def get_datasets(args, transformations=None, shuffle_training=True):
    dataset_class = getattr(datasets, args.train_dataset)
    location = args.data_location

    dataset = dataset_class(
        location=location,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transformations=transformations,
        shuffle_training=shuffle_training,
        fold_number=args.fold_number,
        kwargs=args
    )
    if args.training_split == 'val':
        dataset.train_dataset = dataset.val_dataset
        dataset.train_loader = dataset.val_loader
    elif args.training_split == 'test':
        dataset.train_dataset = dataset.test_dataset
        dataset.train_loader = dataset.test_loader
    if args.test_split == 'val':
        dataset.testtime_dataset = dataset.val_dataset
        dataset.testtime_loader = dataset.val_loader
    elif args.test_split == 'train':
        dataset.testtime_dataset = dataset.train_dataset
        dataset.testtime_loader = dataset.train_loader
    elif args.test_split == 'test':
        dataset.testtime_dataset = dataset.test_dataset
        dataset.testtime_loader = dataset.test_loader
    return dataset


class EntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, reduction='mean', prop_confident_samples=0.1):
        super().__init__()
        self.reduction = reduction
        self.normalization_factor = math.log(num_classes)
        self.prop_confident_samples = prop_confident_samples
        self.forward = self.forward_samples
        # self.forward = self.forward_pixels

    def select_confident_samples(self, logits):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum((1, 2, 3))  # Sum over C, H, W
        idxTPT = torch.argsort(batch_entropy, descending=False)[
                 :int(batch_entropy.size()[0] * self.prop_confident_samples)
                 ]
        return logits[idxTPT]

    def entropy_pixels(self, logits):
        # logits: [N, C, H, W]
        mask = logits.argmax(1)  # N, H, W
        mask = mask.flatten()  # (NHW)
        logits = rearrange(logits, 'n c h w -> c (n h w)')
        batch_entropy = -(logits.softmax(0) * logits.log_softmax(0))

        # Select most confident samples per class
        entropy_classes = []
        for c in range(batch_entropy.size()[0]):  # Iterate over classes
            batch_entropy_c = batch_entropy[c, mask == c]  # (NHW)
            if batch_entropy_c.size()[0] < self.prop_confident_samples * 100:
                # Not enough pixels of this class --> skip
                continue
            selected_idx = torch.argsort(batch_entropy_c, descending=False)[
                           :int(batch_entropy_c.size()[0] * self.prop_confident_samples)
                           ]
            entropy_classes.append(batch_entropy_c[selected_idx].mean())

        entropy_classes = torch.stack(entropy_classes, dim=0)
        return entropy_classes.sum() * (1 / math.log(entropy_classes.size()[0]))

    def avg_entropy_legacy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - self.normalization_factor  # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum()

    def avg_entropy(self, logits):
        c_probs = (logits.softmax(1) * logits.log_softmax(1)).sum(1)  # N, H, W
        hw_probs = - 1 / self.normalization_factor * c_probs.sum((-1, -2))
        return hw_probs.mean()

    @staticmethod
    def entropy(x):
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)  # sum over classes

    def forward_pixels(self, x, *args):
        return self.entropy_pixels(x)

    def forward_samples(self, x, *args):
        if self.prop_confident_samples < 1:
            selected_entropy_maps = self.select_confident_samples(x)
        else:
            selected_entropy_maps = x
        entropy_map = self.avg_entropy(selected_entropy_maps)
        return entropy_map

        # entropy_map = self.entropy(x)
        # entropy_map = entropy_map / self.normalization_factor  # normalize
        # entropy = entropy_map.sum((1, 2))  # sum over pixels
        # if self.reduction == 'mean':
        #     return entropy.mean()
        # return entropy


class IoUHeadLoss(torch.nn.Module):
    def __init__(self, config, reduction='none'):
        super().__init__()
        self.mode = 'multiclass' if config.num_classes > 1 else 'binary'
        self.num_classes = config.num_classes
        self.ignore_index = config.ignore_index
        self.reduction = reduction

    def forward(self, sam_output, target):
        # 1. Compute Pred - Target IoU
        # 2. MSE between predicted IoU and real
        pred = sam_output['masks']
        iou_pred = sam_output['iou_predictions']
        with torch.no_grad():
            tp, fp, fn, tn = metrics.get_stats(pred.argmax(1), target, mode=self.mode, num_classes=self.num_classes,
                                               ignore_index=self.ignore_index)
            iou_real = metrics.iou_score(tp, fp, fn, tn, reduction=self.reduction)
            iou_real = iou_real.to(iou_pred.device)

        mse_loss = torch.nn.functional.mse_loss(iou_pred, iou_real)
        return mse_loss


class MultimaskIoUHeadLoss(torch.nn.Module):
    def __init__(self, reduction='mean', *args):
        super().__init__()
        self.mode = 'binary'
        self.reduction = reduction

    def forward(self, sam_output, target):
        # 1. Compute Pred - Target IoU
        # 2. MSE between predicted IoU and real
        pred = sam_output['multimask_low_res'].sigmoid().unsqueeze(1)
        iou_pred = rearrange(sam_output['iou_predictions'], 'b n -> (b n)')
        with torch.no_grad():
            tp, fp, fn, tn = metrics.get_stats(pred > 0.5, target.unsqueeze(1).long(), mode=self.mode)
            iou_real = metrics.iou_score(tp, fp, fn, tn, reduction='none')[:, 0]
            iou_real = iou_real.to(iou_pred.device)

        mse_loss = torch.nn.functional.mse_loss(iou_pred, iou_real, reduction=self.reduction)
        return mse_loss


class FeedbackLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kl_loss = torch.nn.KLDivLoss()
        self.beta = config.beta_dpo

    # def forward(self, predicted_mask, reference_mask, predicted_iou):
    #     # - L_R + beta * D_KL(ref, predicted)
    #     div = self.kl_loss(predicted_mask.log_softmax(1), reference_mask.softmax(1))
    #     return self.beta * div  + (1 - predicted_iou.mean())

    def forward(self, sam_output, base_logits):
        predicted_mask = sam_output['low_res_logits']
        iou_pred = sam_output['iou_predictions']
        div = self.kl_loss(predicted_mask.log_softmax(1), base_logits.softmax(1))
        return self.beta * div - iou_pred.mean()


def get_loss_fn(args) -> Dict[(str, Callable)]:
    if args.loss_fn is None:
        return {}
    loss_fns = args.loss_fn
    weights = args.loss_weights if args.loss_weights is not None else [1.0] * len(loss_fns)
    assert len(weights) == len(loss_fns), "You must specify a weight for each loss function"

    loss_fn = {}
    for w, l in zip(weights, loss_fns):
        if l in ['DiceLoss', 'JaccardLoss', 'TverskyLoss']:
            fn = getattr(smp.losses, l)(mode='binary' if args.num_classes == 1 else 'multiclass', from_logits=True,
                                        ignore_index=args.ignore_index)
        elif l == 'FocalLoss':
            fn = smp.losses.FocalLoss(mode='binary' if args.num_classes == 1 else 'multiclass',
                                      ignore_index=args.ignore_index)
        elif l == 'CrossEntropyLoss':
            fn = CrossEntropyLoss(ignore_index=args.ignore_index)
        elif l == 'EntropyLoss':
            fn = EntropyLoss(args.num_classes, prop_confident_samples=args.confident_samples_prop)
        elif l == 'IoUHeadLoss':
            fn = IoUHeadLoss(args)
        elif l == 'BinaryDiceLoss':
            fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        elif l == 'FeedbackLoss':
            fn = FeedbackLoss(args)
        elif l == 'MultimaskIoUHeadLoss':
            fn = MultimaskIoUHeadLoss()
        else:
            fn = getattr(smp.losses, l)()

        # loss_fn[l] = lambda y_pred, y_true: float(w) * fn(y_pred, y_true)
        loss_fn[l] = (float(w), fn)
    return loss_fn


def get_model(args, dataset, device='cpu', **kwargs):
    if args.pretrained_model is not None and os.path.isdir(args.pretrained_model):
        # Load the split
        ckpt = os.path.join(args.pretrained_model, f'split_{args.split_number}.pt')
    else:
        ckpt = args.pretrained_model
    # sam = sam_model_registry[args.model](checkpoint=ckpt, num_classes=args.num_classes, image_size=dataset.image_size,
    #                                      **kwargs)
    sam = sam_model_registry[args.model](checkpoint=ckpt, num_classes=args.num_classes,
                                         custom_img_size=dataset.image_size)

    # if args.pretrained_model is not None:
    #     sam.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    #     print(f"\n\tCKPT loaded from {args.pretrained_model}\n")

    if args.baseline == 'lora':
        model = LoRA_Sam(sam, r=4)
        if args.lora_ckpt is not None:
            model.load_lora_parameters(args.lora_ckpt, device)
            print(f"\n\tCKPT loaded from {args.lora_ckpt}\n")

    elif args.baseline == 'decfinetune':
        model = sam
        model.train_encoder = False
        model.disable_unnecesary_gradients()

    elif args.baseline == 'finetune' or args.baseline == '':
        model = sam

    else:
        raise NotImplementedError(f"Baseline {args.baseline} not implemented")

    return model


class ExponentialLRWarmup(torch.optim.lr_scheduler.LRScheduler):
    # Function from SAMed: https://github.com/hitachinsk/SAMed/blob/main/trainer.py
    def __init__(self, optimizer, total_iterations, warmup_iterations, last_epoch=-1):
        self.warmup_iterations = warmup_iterations
        self.total_iterations = total_iterations
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iterations:
            return [base_lr * (self.last_epoch + 1) / self.warmup_iterations for base_lr in self.base_lrs]
        else:
            return [base_lr * (1 - (self.last_epoch - self.warmup_iterations) / self.total_iterations) ** 0.9 for
                    base_lr in self.base_lrs]


def build_optimizer_scheduler(args, model, num_batches):
    param_groups = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd, betas=args.betas)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * num_batches)
    elif args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs * num_batches, gamma=1)
    elif args.scheduler == 'exponential':
        scheduler = ExponentialLRWarmup(optimizer, args.epochs * num_batches, args.warmup_length)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented")

    return optimizer, scheduler


def make_name(args, current_time):
    name = args.train_dataset
    if args.task == 'training_encoder':
        name += '_enc'
    elif args.task == 'training':
        pass
    elif args.task == 'lora':
        name += '_lora'
    elif args.task == 'test_time_volume':
        name += '_ttv'
    elif args.task == 'sfda_rl':
        name += '_sfda'
    elif args.task == 'test_time_rl':
        name += '_ttrl'
    elif args.task == 'reward_correlation':
        name += '_rc'
    elif args.task == 'testing':
        name += '_test'
    elif args.task == 'training_mask_prompting':
        name += '_mp'

    name += f'_{args.baseline}' if args.baseline != '' else ''

    name += '_' + current_time
    return name


def build_testing_savepath(args):
    save_path = os.path.join(args.save, args.exp_name)
    i = 1
    while os.path.exists(save_path):
        save_path = '_'.join(save_path.split('_')[:-1])
        save_path = save_path + '_' + str(i)
        i += 1
    os.makedirs(save_path)
    return save_path


def prepare_intermediate_rep(x):
    x = torch.nn.functional.avg_pool2d(x, *(x.shape[-2:]))
    x = x.flatten(start_dim=1)
    return x


def compute_class_weights(args, target, pixels_class):
    class_weights = [target.nelement() / (pixels_class[i]) if pixels_class[i] != 0 else 0 for i in
                     range(args.num_classes)]
    return torch.tensor(class_weights)


def dict_to_list(dict_to, name, dict_values):
    for key in dict_values.keys():
        if key not in dict_to.keys():
            dict_to[key] = {name: [dict_values[key]]}
            continue
        if name not in dict_to[key].keys():
            dict_to[key][name] = [dict_values[key]]
        else:
            dict_to[key][name].append(dict_values[key])
    return dict_to


def prepare_batch(args, batch, device='cpu'):
    inp = batch['image'].to(device)
    original_size = inp.shape[-2:]  # They should all have the same shape

    if args.prompting:
        if random.random() < 0.5:
            boxes = batch['boxes'].to(device)
            point = None
            target = batch['mask_bb'].to(device)
            low_res_target = batch['mask_bb_downsampled'].to(device)
        else:
            boxes = None
            point = (batch['point'].to(device), batch['point_label'].to(device))
            target = batch['mask_point'].to(device)
            low_res_target = batch['mask_point_downsampled'].to(device)
    else:
        boxes, point = None, None
        target = batch['mask'].to(device)
        low_res_target = batch['mask_downsampled'].to(device)

    return [inp, original_size, boxes, point], target, low_res_target


def write_results(args, results, name):
    results = {k: v for d in results for k, v in d.items()}
    for k in results.keys():
        df = pd.DataFrame.from_dict(results[k], orient='index')
        df.to_csv(os.path.join(args.save, f'{name}_{k}.csv'))


def project_name_composer(args):
    name = 'SAM_EUROPA'
    ttda = ['test_time', 'test_time_rl', 'test_time_volume']
    if args.wandb_project != '':
        return args.wandb_project
    if args.task in ttda:
        name += '_TTDA'
    elif args.task == 'train_reward':
        name += '_RW'
    elif args.task == 'sfda_rl':
        name += '_SFDA'
    elif args.task == 'dpo':
        name += '_DPO'
    return name


def update_stats(stats, tp, fp, fn, tn):
    stats['tp'].append(tp)
    stats['fp'].append(fp)
    stats['fn'].append(fn)
    stats['tn'].append(tn)
    return stats


def compute_iou_stats(args, stats_evolution):
    iou_iterations = []
    for i, iteration_stats in enumerate(stats_evolution):
        tp, fp, fn, tn = iteration_stats.values()
        tp = torch.cat(tp, dim=0)
        fp = torch.cat(fp, dim=0)
        fn = torch.cat(fn, dim=0)
        tn = torch.cat(tn, dim=0)

        iou = metrics.iou_score(tp, fp, fn, tn, reduction='macro')
        iou_iterations.append(iou)
        print(f"IOU it {i + 1}: {iou}")

    if args.wandb:
        wandb.log({'iou': iou})
    return iou, iou_iterations


def torch_distributed_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '27883'

    # Initialize the process group
    # For some reason the backend "nccl" freezes for me, but "gloo" works.
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def torch_distributed_cleanup():
    torch.distributed.destroy_process_group()


def save_image(mask, input_image, dataset, save_path):
    from PIL import Image
    import torch.nn.functional as F
    from torchvision.utils import draw_segmentation_masks

    mask = mask.argmax(1, keepdim=True)

    oh_mask = F.one_hot(mask[0], dataset.num_classes)[0].permute(2, 0, 1).bool()
    oh_mask[0] = torch.zeros_like(oh_mask[0])
    save_img = draw_segmentation_masks(input_image[0], oh_mask, alpha=0.5,
                                       colors=['#7FAF22', '#217FAF', '#AF217F', '#7FAF21'])
    save_img = Image.fromarray(save_img.permute(1, 2, 0).detach().cpu().numpy())
    save_img.save(save_path)


def point_from_mask(input_mask, previous_prediction, previous_points, device):
    '''
    input_mask: [N, H, W]
    previous_prediction: [N, C, H, W]
    previous_points: [N, X, 2]
    '''
    new_points = []
    labels = []
    # Compute the error region. Only the foreground parts:
    error_region = ((input_mask != previous_prediction.argmax(1)) * (input_mask != 0)).float()
    # Sample a point from the error region tensor.
    for i in range(input_mask.size(0)):
        error_indices = torch.nonzero(error_region[i], as_tuple=False)
        if error_indices.size(0) == 0:
            random_point = previous_points[i, -1:]
        else:
            random_point = error_indices[torch.randint(0, error_indices.size(0), (1,))]
            # These are (y, x). We have to transform them to (x, y)
            random_point = random_point.flip(0)

        new_points.append(random_point.tolist())

        # Get the label of the point. 1 if it's a false negative, 0 if it's a false positive.
        # It will always be a false negative because we are forcing it before. TODO: Sample with probability
        # if input_mask[i, random_point[0, 0], random_point[0, 1]] == 0:
        #     labels.append([0])
        # else:
        #     labels.append([1])
        labels.append([1])

    return torch.tensor(new_points).to(device), torch.tensor(labels).to(device)


if __name__ == '__main__':
    mask = torch.randint(0, 2, (2, 512, 512))
    previous_prediction = torch.randint(0, 2, (2, 2, 512, 512))
    previous_points = torch.tensor([[0, 0], [0, 0]])
    point_from_mask(mask, previous_prediction, previous_points)
