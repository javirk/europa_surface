# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
import torch.nn.functional as F

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT


def build_sam_vit_h(checkpoint=None, custom_img_size=1024, num_classes=3):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        custom_img_size=custom_img_size,  # by LBK EDIT
        num_classes=num_classes,
    )


def build_sam_vit_l(checkpoint=None, custom_img_size=1024, num_classes=3):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        custom_img_size=custom_img_size,  # by LBK EDIT
        num_classes=num_classes,
    )


# by LBK EDIT
def build_sam_vit_b(checkpoint=None, custom_img_size=1024, num_classes=3):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        custom_img_size=custom_img_size,  # by LBK EDIT
        num_classes=num_classes,
    )


# Original SAM
def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
        custom_img_size=1024,  # by LBK EDIT
        num_classes=3,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=custom_img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            # LBK EDIT (Important)
            # image_embedding_size=(image_embedding_size, image_embedding_size),
            # input_image_size=(image_size, image_size),
            image_embedding_size=(custom_img_size // vit_patch_size, custom_img_size // vit_patch_size),
            input_image_size=(custom_img_size, custom_img_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, custom_img_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)
    return sam


# Mobile-SAM
def build_sam_vit_t(checkpoint=None, custom_img_size=1024):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
        image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                              embed_dims=[64, 128, 160, 320],
                              depths=[2, 2, 6, 2],
                              num_heads=[2, 4, 5, 10],
                              window_sizes=[7, 7, 14, 7],
                              mlp_ratio=4.,
                              drop_rate=0.,
                              drop_path_rate=0.0,
                              use_checkpoint=False,
                              mbconv_expand_ratio=4.0,
                              local_conv_size=3,
                              layer_lr_decay=0.8
                              ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            # LBK EDIT (Important)
            # image_embedding_size=(image_embedding_size, image_embedding_size),
            # input_image_size=(image_size, image_size),
            image_embedding_size=(custom_img_size // vit_patch_size, custom_img_size // vit_patch_size),
            input_image_size=(custom_img_size, custom_img_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        mobile_sam.load_state_dict(state_dict)
    return mobile_sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
}


def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head', 'blocks_adapt']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[
                          2] not in k and except_keys[3] not in k}

    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            if k not in new_state_dict.keys():  # The encoder adapter will fall here, but we won't need to load it
                continue
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear',
                                           align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict
