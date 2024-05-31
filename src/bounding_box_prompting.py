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
from segment_anything import SamBBoxMaskGenerator
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


def bounding_box_prompt(args):
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
        dataset = dataset_class(root=args.data_location, split='test', fold_number=0, get_full_sample=True)
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

            mask_generator = SamBBoxMaskGenerator(model)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.red())

            for k, data in tqdm(enumerate(dataset)):
                inp = data['image'].permute(1,2,0).numpy()  # [3, H, W]
                name = data['name']
                boxes = data['bboxes']  # [N, 4]

                # Batch the boxes first
                sam_result = mask_generator.generate(inp, boxes)
                # "segmentation": mask_data['masks'],
                # "bbox": mask_data['boxes'],
                # "predicted_iou": mask_data['iou_preds'],

                mask = sam_result['segmentation']  # Numpy array already

                instances = (mask[:, 1:].sum(axis=1) > 0)

                instance_detections = sv.Detections(xyxy=boxes.numpy(), mask=instances, class_id=np.zeros(len(boxes)))
                annotated_image_semantic = mask_annotator.annotate(scene=inp.copy(),
                                                                   detections=instance_detections)
                inp = box_annotator.annotate(scene=inp.copy(), detections=instance_detections)

                plot_images_grid(
                    images=[inp, annotated_image_semantic],
                    grid_size=(1, 2),
                    titles=['source image', 'instance segmentation'],
                    save_path=os.path.join(args.save, name + '.png')
                )

                # detections = sv.Detections.from_sam(sam_result=sam_result)  # This does not use the class id
                # sem_mask = np.zeros([5, *mask.shape[1:]])
                #
                # # Put all the annotations together for semantic segmentation
                # for j in range(len(mask)):
                #     sem_mask[labels[j], mask[j] == 1] = 1
                # semantic_detections = sv.Detections(xyxy=np.zeros((sem_mask.shape[0], 4)), mask=sem_mask.astype(bool))
                # annotated_image = mask_annotator.annotate(scene=inp.copy(),
                #                                           detections=detections)  # Instance segmentation
                # annotated_image_semantic = mask_annotator.annotate(scene=inp.copy(),
                #                                                    detections=semantic_detections)  # Semantic seg
                #
                # plot_images_grid(
                #     images=[inp, annotated_image, annotated_image_semantic],
                #     grid_size=(1, 3),
                #     titles=['source image', 'instance segmentation', 'semantic segmentation'],
                #     save_path=os.path.join(args.save, name + '.png')
                # )
