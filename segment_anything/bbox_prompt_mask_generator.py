# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    does_mask_contain_point,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    is_box_empty,
)


class SamBBoxMaskGenerator:
    def __init__(
        self,
        model: Sam,
        bboxes_per_batch: int = 64 // 4,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM models, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM models to use for mask prediction.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401


        self.predictor = SamPredictor(model)
        self.bboxes_per_batch = bboxes_per_batch
        self.output_mode = output_mode
        self.crop_n_layers = 0
        self.crop_overlap_ratio = 512 / 1500

    @torch.no_grad()
    def generate(self, images: [torch.Tensor, np.ndarray], bboxes: [torch.Tensor, List]) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          images (torch.Tensor): The images to generate masks for, in BCHW uint8 format.
                 (np.ndarray): The images to generate masks for, in HWC uint8 format.
          bboxes (torch.Tensor): The bounding boxes to generate masks for, in XYXY format. BN4

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The models's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the models to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        if isinstance(images, np.ndarray):
            images = torch.as_tensor(images)
            if images.dim() == 3:
                images = images.permute(2, 0, 1).unsqueeze(0)

        if not isinstance(bboxes, list):
            bboxes = [bboxes]

        # Generate masks
        mask_data = self._generate_masks(images, bboxes)

        ann = {
            "segmentation": mask_data['thresholded_masks'],
            "logits_mask": mask_data['masks'],
            "bbox": mask_data['boxes'],
            "predicted_iou": mask_data['iou_preds'],
        }

        return ann

    def _generate_masks(self, images: torch.Tensor, bounding_boxes: torch.Tensor) -> MaskData:
        orig_size = images.shape[-2:]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(images, crop_box, layer_idx, orig_size, bounding_boxes)
            data.cat(crop_data)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        images: torch.Tensor,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        bounding_boxes: torch.Tensor
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = images[:, :, y0:y1, x0:x1]
        cropped_im_size = cropped_im.shape[-2:]
        self.predictor.set_images_batch(cropped_im)

        # Generate masks for this crop in batches
        data = MaskData()
        for (boxes,) in batch_iterator(self.bboxes_per_batch, bounding_boxes):
            batch_data = self._process_batch(boxes, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        return data

    def _process_batch(
        self,
        boxes: torch.Tensor,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run models on this batch
        in_boxes = self.predictor.transform.apply_boxes_torch_batch(boxes, im_size)
        masks, iou_preds, _ = self.predictor.predict_torch_batch(
            boxes=in_boxes,
            multimask_output=True,
            return_logits=True,
        )

        thresholded_masks = [mask > 0.0 for mask in masks]

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks,
            thresholded_masks=thresholded_masks,
            iou_preds=iou_preds,
            boxes=boxes,
        )

        return data
