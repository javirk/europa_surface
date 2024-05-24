import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
import supervision as sv
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from supervision.annotators.base import ImageType
from supervision.utils.conversion import pillow_to_cv2

import src.datasets as datasets
from src.models.utils import get_model
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything.utils.amg import box_xywh_to_xyxy


def plot_images_grid(
        images: List[ImageType],
        grid_size: Tuple[int, int],
        titles: Optional[List[str]] = None,
        size: Tuple[int, int] = (12, 12),
        cmap: Optional[str] = "gray",
        save_path: Optional[str] = None
) -> None:
    """
    Plots images in a grid using matplotlib.

    Args:
       images (List[ImageType]): A list of images as ImageType
             is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
       grid_size (Tuple[int, int]): A tuple specifying the number
            of rows and columns for the grid.
       titles (Optional[List[str]]): A list of titles for each image.
            Defaults to None.
       size (Tuple[int, int]): A tuple specifying the width and
            height of the entire plot in inches.
       cmap (str): the colormap to use for single channel images.

    Raises:
       ValueError: If the number of images exceeds the grid size.
    """
    nrows, ncols = grid_size

    for idx, img in enumerate(images):
        if isinstance(img, Image.Image):
            images[idx] = pillow_to_cv2(img)

    if len(images) > nrows * ncols:
        raise ValueError(
            "The number of images exceeds the grid size. Please increase the grid size"
            " or reduce the number of images."
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap=cmap)
            else:
                ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])

        ax.axis("off")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def instance_seg(args):
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
            args.lora_ckpt = fold_ckpt
            model = get_model(args, dataset)

            if args.testing_ckpt:
                model.load_state_dict(torch.load(fold_ckpt, map_location=device))

            # model = torch.nn.DataParallel(model, device_ids=devices)
            model.to(device)
            model.eval()

            mask_generator = SamAutomaticMaskGenerator(model, pred_iou_thresh=0.25, min_mask_region_area=200,
                                                       points_per_side=8)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

            for i, data in tqdm(enumerate(dataset)):
                inp = data['image']
                name = data['name']
                # Input must be in HWC uint8 format
                inp = inp.permute(1, 2, 0).numpy().astype('uint8')
                sam_result = mask_generator.generate(inp)

                sorted_generated_masks = sorted(
                    sam_result, key=lambda x: x["predicted_iou"], reverse=True
                )

                mask = np.array([mask["segmentation"] for mask in sorted_generated_masks])
                labels = np.array([mask['class_id'] for mask in sorted_generated_masks])
                logits_mask = np.array([mask['logits_mask'] for mask in sorted_generated_masks])
                bbox = np.array([box_xywh_to_xyxy(mask['bbox']) for mask in sorted_generated_masks])
                scores = np.array([min(1, mask['predicted_iou']) for mask in sorted_generated_masks])

                # Save masks, bboxes, labels and scores into hdf5py
                with h5py.File(os.path.join(args.save, name + '.hdf5'), "w") as f:
                    f.create_dataset("mask", data=mask)
                    f.create_dataset("labels", data=labels)
                    f.create_dataset("logits_mask", data=logits_mask)
                    f.create_dataset("bbox", data=bbox)
                    f.create_dataset("scores", data=scores)

                detections = sv.Detections.from_sam(sam_result=sam_result)  # This does not use the class id
                sem_mask = np.zeros([5, *mask.shape[1:]])

                # Put all the annotations together for semantic segmentation
                for j in range(len(mask)):
                    sem_mask[labels[j], mask[j] == 1] = 1
                semantic_detections = sv.Detections(xyxy=np.zeros((sem_mask.shape[0], 4)), mask=sem_mask.astype(bool))
                annotated_image = mask_annotator.annotate(scene=inp.copy(),
                                                          detections=detections)  # Instance segmentation
                annotated_image_semantic = mask_annotator.annotate(scene=inp.copy(),
                                                                   detections=semantic_detections)  # Semantic seg

                plot_images_grid(
                    images=[inp, annotated_image, annotated_image_semantic],
                    grid_size=(1, 3),
                    titles=['source image', 'instance segmentation', 'semantic segmentation'],
                    save_path=os.path.join(args.save, name + '.png')
                )
