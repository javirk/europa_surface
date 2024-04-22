# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from collections import OrderedDict

import torch.nn.functional as F
from typing import Any, List, Mapping, Optional
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam


class NoEncoderSam(Sam): # TODO: Modify SAM forward
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__(image_encoder=None, prompt_encoder=prompt_encoder, mask_decoder=mask_decoder,
                         pixel_mean=pixel_mean, pixel_std=pixel_std)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            mask: torch.Tensor,
            boxes: Optional[torch.Tensor] = None,
            points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the models directly.

        Arguments:
          image_embeddings (torch.Tensor): The embeddings from the encoder.
          mask (torch.Tensor): The ground truth mask for the image.
          boxes (torch.Tensor): Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the models.
          points (torch.Tensor): Batched point inputs, with shape Bx2.

        Returns:
          (torch.Tensor): Batched binary mask predictions,
            with shape BxCxHxW, where B is the number of input prompts,
            C is determined by multimask_output, and (H, W) is the
            original size of the image.
        """
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            # if len(boxes) == 2:
            #     boxes = boxes[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

    def load_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        state_dict = OrderedDict(state_dict)
        for k in list(state_dict.keys()):
            if k.startswith("image_encoder."):
                del state_dict[k]

        self.load_state_dict(state_dict, strict=strict)
