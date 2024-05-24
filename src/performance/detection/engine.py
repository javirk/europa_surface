import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import numpy as np

import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1.0 / 1000 # start_factor (float):
    #     # The number we multiply learning rate in the first epoch.
    #     # The multiplication factor changes towards end_factor in the following epochs.
    #     # Default: 1./3.
    #     warmup_iters = min(1000, len(data_loader) - 1)

    #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #         optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    #     )

    idx=0
    # following goes through one batch
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        idx += 1
        # CH: below isn't necessary anymore since I filter out empty images directly
        # print(idx)
        # make list of images
        images = list(image for image in images)
        # get rid of images with empty masks/boxes
        # indices = np.arange(0, len(images), 1)
        # for i, target in enumerate(targets):
        #     if target == None:
        #         print('empty target identified, getting rid of this one')
        #         # delete target by masking targets with 'not-target'
        #         np.delete(indices, i)
        # print(indices)
        # print(indices.dtype)
        # print(type(indices))

        # get rid of images with empty masks/boxes
        images = [image for image in images if image != None]
        targets = [target for target in targets if target != None]

        # images = images[indices] # torch.index_select(images, 0, torch.as_tensor(indices))
        # targets = targets[indices] # torch.index_select(targets, 0, torch.as_tensor(indices))
        # targets = targets[del_indices]
        # images = images[~del_indices]
        # idx = idx+1

        key_list = ["boxes", "labels", "masks", "image_id", "area", "iscrowd"]

        # for target in targets:
        #     print(target['boxes'])
        #     print(target['boxes'].shape)
        #     for key, value in target.items():
        #         if key in key_list:
        #             try:
        #                 value.to(device)
        #             except AttributeError:
        #                 print(key, value)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k in key_list} for t in targets ]

        # # for debugging #############3
        # print('\nTRAINING')
        # print('len(images) = {}'.format(len(images)))
        # image = images[0]
        # target = targets[0]
        # print('image dtype: {}'.format(image.dtype))
        # print('image shape: {}'.format(image.shape))
        # print('image histogram: {}'.format(np.histogram(image.cpu())))
        # # target: masks
        # print('masks dtype: {}'.format(target['masks'].dtype))
        # print('masks shape: {}'.format(target['masks'].shape))
        # print('masks histogram: {}'.format(np.histogram(target['masks'].cpu())))
        # # target: bbox
        # print('bbox dtype: {}'.format(target['boxes'].dtype))
        # print('bbox shape: {}'.format(target['boxes'].shape))
        # print('bbox histogram: {}'.format(np.histogram(target['boxes'].cpu())))
        # # target: labels
        # print('label dtype: {}'.format(target['labels'].dtype))
        # print('label shape: {}'.format(target['labels'].shape))
        # print('labels: {}'.format(target['labels'].cpu())) 
        ###########################

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # print('image length is: {}'.format(len(images)))
            # print(images)
            # print('targets length is {}'.format(len(targets)))
            # print(targets)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        # if lr_scheduler is not None:
        #     lr_scheduler.step()


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if writer != None:
            writer.add_scalar("training total loss", loss_value, idx + epoch*len(data_loader))
            writer.add_scalar("training mask loss", loss_dict_reduced['loss_mask'], idx + epoch*len(data_loader))
            writer.add_scalar("training box regression loss", loss_dict_reduced['loss_box_reg'], idx + epoch*len(data_loader))
            writer.add_scalar("training rpn box regression loss", loss_dict_reduced['loss_rpn_box_reg'], idx + epoch*len(data_loader))
            writer.add_scalar("training objectness loss", loss_dict_reduced['loss_objectness'], idx + epoch*len(data_loader))

        # return loss, lr and loss_dict_reduced in own dictionary,
        # since I am not sure how to use metric_logger

        # # debugging, after first loop
        # from IPython.core.debugger import set_trace
        # set_trace()

    return loss_value, loss_dict_reduced


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, epochidx=0, writer=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset) # this returns an object <pycocotools.coco.COCO at 0x1f12153c3d0>
    iou_types = _get_iou_types(model) # CH: this looks like ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)
    # change parameters in CocoEvaluator

    # CH on 2022-05-10, for my mAP calculation
    predlabels = []
    predscores = []
    predboxes = []
    predmasks = []

    targetboxes = []
    targetlabels = []
    targetmasks = []

    rec_thrs = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
        0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
        0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
        0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
        0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
        0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
        0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
        0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
        0.99, 1.  ])

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

               
        # # for debugging #############
        # print('len(images) = {}'.format(len(images)))
        # image = images[0]
        # target = targets[0]
        # output = outputs[0]
        # print('image dtype: {}'.format(image.dtype))
        # print('image shape: {}'.format(image.shape))
        # print('image histogram: {}'.format(np.histogram(image.cpu())))
        # # target: masks
        # print('masks dtype: {}'.format(target['masks'].dtype))
        # print('masks shape: {}'.format(target['masks'].shape))
        # print('masks histogram: {}'.format(np.histogram(target['masks'].cpu())))
        # # target: bbox
        # print('bbox dtype: {}'.format(target['boxes'].dtype))
        # print('bbox shape: {}'.format(target['boxes'].shape))
        # print('bbox histogram: {}'.format(np.histogram(target['boxes'].cpu())))
        # # target: labels
        # print('label dtype: {}'.format(target['labels'].dtype))
        # print('label shape: {}'.format(target['labels'].shape))
        # print('labels: {}'.format(target['labels'].cpu())) 
        # # output
        # print('\nOUTPUTS')
        # # outputs: masks
        # print('outputs masks dtype: {}'.format(output['masks'].dtype))
        # print('outputs masks shape: {}'.format(output['masks'].shape))
        # print('outputs masks histogram: {}'.format(np.histogram(output['masks'].cpu())))
        # # outputs: bbox
        # print('outputs bbox dtype: {}'.format(output['boxes'].dtype))
        # print('outputs bbox shape: {}'.format(output['boxes'].shape))
        # print('outputs bbox histogram: {}'.format(np.histogram(output['boxes'].cpu())))     
        # # outputs: labels
        # print('outputs label dtype: {}'.format(output['labels'].dtype))
        # print('outputs label shape: {}'.format(output['labels'].shape))
        # print('outputs labels: {}'.format(output['labels'].cpu()))   
        # ###########################

        # CH, 2022-05-04, I thought I've add mean average precision per class
        # mappreds = [{'scores': y_scores[idx].detach(), 'labels': y_preds[idx]} for idx in range(len(y_scores))] # note that len(y_scores[0]) = batch_size
        # maptarget = [{'labels': y_true[idx]} for idx in range(len(y_true))]
        # metric = MeanAveragePrecision()
        # metric.update(mappreds, maptarget)
        # mAP = metric.compute()

        # loop through batch and append to metric lists
        for idx, target in enumerate(targets):
            # first, append target lists
            targetlabels.append(target['labels'].detach().cpu())
            targetboxes.append(target['boxes'].detach().cpu())
            targetmasks.append(target['masks'].detach().cpu())

            # get predictions from 'outputs'
            predlabels.append(outputs[idx]['labels'].detach().cpu())
            predscores.append(outputs[idx]['scores'].detach().cpu())
            predboxes.append(outputs[idx]['boxes'].detach().cpu())
            predmasks.append(outputs[idx]['masks'].detach().cpu())

    mappreds = [{'boxes': predboxes[idx], 'scores': predscores[idx], 'labels': predlabels[idx]} for idx in range(len(predlabels)) ] # note that len(y_scores[0]) = batch_size
    maptarget = [{'boxes': targetboxes[idx], 'labels': targetlabels[idx]} for idx in range(len(predlabels)) ]
    # print('mappreds: {}'.format(len(mappreds)))
    # print('maptargets: {}'.format(len(maptarget)))
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(mappreds, maptarget)
    mdict = metric.compute()
    print(mdict)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize() # --> this is how we can output, but how can we get a dictionary or similar?
    torch.set_num_threads(n_threads)

    if writer != None:
        writer.add_scalar("validation mAP", mdict['map'], epochidx) # 3023-07-06_ changed idx + epochidx to epochidx (other is wrong)
        writer.add_scalar("validation mAP_50", mdict['map_50'], epochidx) # 3023-07-06: added map_50
        try:
            for i, classmap in enumerate(mdict['map_per_class']):
                writer.add_scalar("validation mAP of class " + str(i), classmap.item(), epochidx)

        except TypeError:
            print('could not print per class mAP.')

        # 2023-07-06: add precision and recall for mask and box
        # copied and adjusted from 'Z:\Groups\PIG\Caroline\lineament_detection\pytorch_maskrcnn\code\galileo_dataset_pub2_nocusps\pub_files\code\Precision_recall.py'
        # rec_thrs is defined outside loop
        for ioutype in iou_types:
            precision = coco_evaluator.coco_eval[ioutype].eval['precision']
            recall = coco_evaluator.coco_eval[ioutype].eval['recall'] # is of shape (10, 4, 4, 3)
            scores = coco_evaluator.coco_eval[ioutype].eval['scores'] # is of shape (10, 101, 4, 4, 3)
            t_ind=1 # t_ind=0--> IoU=0.35, t_ind=1 --> IoU=0.5
            # or IoU threshold index = 0 if iou = 0.35
            a_ind = 0 # 0 --> 'all'
            m_ind = 2 # 2 --> 100 max detections
            precision = precision[t_ind, :, :, a_ind, m_ind] # now the shape is (101, 4)
            scores = scores[t_ind, :, :, a_ind, m_ind] # now the shape is (101, 4)
            recall = recall[t_ind, :, a_ind, m_ind] # average recall at IoU=0.5 
            # second index would be the cat_indx
            for cat_idx in range(precision.shape[1]): 
                try:
                    # output precision and recall for score >= 0.5
                    print('{} precision of class {} at score of 0.5 and IoU=0.5: {}'.format(ioutype, cat_idx, precision[:,cat_idx][scores[:,cat_idx]>=0.5].min()))
                    print('{} recall of class {} at score of 0.5 and IoU=0.5: {}'.format(ioutype, cat_idx, rec_thrs[scores[:,cat_idx]>=0.5].max()))
                    # print(precision.shape)
                    # print(precision)
                    writer.add_scalar("{} precision of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), precision[:,cat_idx][scores[:,cat_idx]>=0.5].min(), epochidx)
                    writer.add_scalar("{} recall of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), rec_thrs[scores[:,cat_idx]>=0.5].max(), epochidx)
                except ValueError: # then, the .min() raises an error because the array is simply empty
                    writer.add_scalar("{} precision of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), 0, epochidx)
                    writer.add_scalar("{} recall of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), 0, epochidx)                    

    return mdict


# CH on 2024-05-22
# adapted method for evaluating SAM
@torch.inference_mode()
def evaluate_fromoutput(model_output, data_loader, device, epochidx=0, writer=None):
    '''
    Difference to method above is that a model output in the form
    model_output = [
    {
    'boxes': torch.tensor(shape=(N, 4)), 
    'labels': torch.tensor(shape=(N,)), 
    'scores': torch.tensor(shape=(N,)), 
    'masks': torch.tensor(shape=(N, 1, W, H))
    },
    {...},
    {...},
    ...
    ]
    (a list of dictionaries)

    Can be passed directly.
    '''
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset) # this returns an object <pycocotools.coco.COCO at 0x1f12153c3d0>
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)
    # change parameters in CocoEvaluator

    # CH on 2022-05-10, for my mAP calculation
    predlabels = []
    predscores = []
    predboxes = []
    predmasks = []

    targetboxes = []
    targetlabels = []
    targetmasks = []

    rec_thrs = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
        0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
        0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
        0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
        0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
        0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
        0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
        0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
        0.99, 1.  ])

    # CH: i changed the next line to enumerate
    for imgidx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        images = list(img.to(device) for img in images)
        # print('images has length: {}'.format(len(images)))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        # outputs = model(images)
        # above line commented, instead, we get model output directly from the function input
        # indexed with image index, batch size and length of the current loaded images (might be smaller than the batch size)
        outputs = model_output[imgidx*data_loader.batch_size : imgidx*data_loader.batch_size+len(images)]

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

               
        # # for debugging #############
        # print('len(images) = {}'.format(len(images)))
        # image = images[0]
        # target = targets[0]
        # output = outputs[0]
        # print('image dtype: {}'.format(image.dtype))
        # print('image shape: {}'.format(image.shape))
        # print('image histogram: {}'.format(np.histogram(image.cpu())))
        # # target: masks
        # print('masks dtype: {}'.format(target['masks'].dtype))
        # print('masks shape: {}'.format(target['masks'].shape))
        # print('masks histogram: {}'.format(np.histogram(target['masks'].cpu())))
        # # target: bbox
        # print('bbox dtype: {}'.format(target['boxes'].dtype))
        # print('bbox shape: {}'.format(target['boxes'].shape))
        # print('bbox histogram: {}'.format(np.histogram(target['boxes'].cpu())))
        # # target: labels
        # print('label dtype: {}'.format(target['labels'].dtype))
        # print('label shape: {}'.format(target['labels'].shape))
        # print('labels: {}'.format(target['labels'].cpu())) 
        # # output
        # print('\nOUTPUTS')
        # # outputs: masks
        # print('outputs masks dtype: {}'.format(output['masks'].dtype))
        # print('outputs masks shape: {}'.format(output['masks'].shape))
        # print('outputs masks histogram: {}'.format(np.histogram(output['masks'].cpu())))
        # # outputs: bbox
        # print('outputs bbox dtype: {}'.format(output['boxes'].dtype))
        # print('outputs bbox shape: {}'.format(output['boxes'].shape))
        # print('outputs bbox histogram: {}'.format(np.histogram(output['boxes'].cpu())))     
        # # outputs: labels
        # print('outputs label dtype: {}'.format(output['labels'].dtype))
        # print('outputs label shape: {}'.format(output['labels'].shape))
        # print('outputs labels: {}'.format(output['labels'].cpu()))   
        # ###########################

        # CH, 2022-05-04, I thought I've add mean average precision per class
        # mappreds = [{'scores': y_scores[idx].detach(), 'labels': y_preds[idx]} for idx in range(len(y_scores))] # note that len(y_scores[0]) = batch_size
        # maptarget = [{'labels': y_true[idx]} for idx in range(len(y_true))]
        # metric = MeanAveragePrecision()
        # metric.update(mappreds, maptarget)
        # mAP = metric.compute()

        # loop through batch and append to metric lists
        for idx, target in enumerate(targets):
            # first, append target lists
            targetlabels.append(target['labels'].detach().cpu())
            targetboxes.append(target['boxes'].detach().cpu())
            targetmasks.append(target['masks'].detach().cpu())

            # get predictions from 'outputs'
            predlabels.append(outputs[idx]['labels'].detach().cpu())
            predscores.append(outputs[idx]['scores'].detach().cpu())
            predboxes.append(outputs[idx]['boxes'].detach().cpu())
            predmasks.append(outputs[idx]['masks'].detach().cpu())

    mappreds = [{'boxes': predboxes[idx], 'scores': predscores[idx], 'labels': predlabels[idx]} for idx in range(len(predlabels)) ] # note that len(y_scores[0]) = batch_size
    maptarget = [{'boxes': targetboxes[idx], 'labels': targetlabels[idx]} for idx in range(len(predlabels)) ]
    # print('mappreds: {}'.format(len(mappreds)))
    # print('maptargets: {}'.format(len(maptarget)))
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(mappreds, maptarget)
    mdict = metric.compute()
    print(mdict)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize() # --> this is how we can output, but how can we get a dictionary or similar?
    torch.set_num_threads(n_threads)

    if writer != None:
        writer.add_scalar("validation mAP", mdict['map'], epochidx) # 3023-07-06_ changed idx + epochidx to epochidx (other is wrong)
        writer.add_scalar("validation mAP_50", mdict['map_50'], epochidx) # 3023-07-06: added map_50
        try:
            for i, classmap in enumerate(mdict['map_per_class']):
                writer.add_scalar("validation mAP of class " + str(i), classmap.item(), epochidx)

        except TypeError:
            print('could not print per class mAP.')

        # 2023-07-06: add precision and recall for mask and box
        # copied and adjusted from 'Z:\Groups\PIG\Caroline\lineament_detection\pytorch_maskrcnn\code\galileo_dataset_pub2_nocusps\pub_files\code\Precision_recall.py'
        # rec_thrs is defined outside loop
        for ioutype in iou_types:
            precision = coco_evaluator.coco_eval[ioutype].eval['precision']
            recall = coco_evaluator.coco_eval[ioutype].eval['recall'] # is of shape (10, 4, 4, 3)
            scores = coco_evaluator.coco_eval[ioutype].eval['scores'] # is of shape (10, 101, 4, 4, 3)
            t_ind=1 # t_ind=0--> IoU=0.35, t_ind=1 --> IoU=0.5
            # or IoU threshold index = 0 if iou = 0.35
            a_ind = 0 # 0 --> 'all'
            m_ind = 2 # 2 --> 100 max detections
            precision = precision[t_ind, :, :, a_ind, m_ind] # now the shape is (101, 4)
            scores = scores[t_ind, :, :, a_ind, m_ind] # now the shape is (101, 4)
            recall = recall[t_ind, :, a_ind, m_ind] # average recall at IoU=0.5 
            # second index would be the cat_indx
            for cat_idx in range(precision.shape[1]): 
                try:
                    # output precision and recall for score >= 0.5
                    print('{} precision of class {} at score of 0.5 and IoU=0.5: {}'.format(ioutype, cat_idx, precision[:,cat_idx][scores[:,cat_idx]>=0.5].min()))
                    print('{} recall of class {} at score of 0.5 and IoU=0.5: {}'.format(ioutype, cat_idx, rec_thrs[scores[:,cat_idx]>=0.5].max()))
                    # print(precision.shape)
                    # print(precision)
                    writer.add_scalar("{} precision of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), precision[:,cat_idx][scores[:,cat_idx]>=0.5].min(), epochidx)
                    writer.add_scalar("{} recall of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), rec_thrs[scores[:,cat_idx]>=0.5].max(), epochidx)
                except ValueError: # then, the .min() raises an error because the array is simply empty
                    writer.add_scalar("{} precision of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), 0, epochidx)
                    writer.add_scalar("{} recall of class {} (IoU=0.5, score=0.5)".format(ioutype, cat_idx), 0, epochidx)                    

    return mdict