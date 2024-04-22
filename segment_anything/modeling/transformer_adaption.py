# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

from typing import Tuple, Type, Optional

from .transformer import Attention, TwoWayAttentionBlock
from .image_encoder import PatchEmbed, Block, LayerNorm2d
from .image_encoder import Attention as ImageAttention


class TwoWayTransformerAdaption(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            adapter_len: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.adapter_len = adapter_len
        self.layers = nn.ModuleList()
        self.layers_adapt = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
            self.layers_adapt.append(
                nn.ModuleDict({
                    # "w_adapt": nn.Linear(embedding_dim * 2, embedding_dim, bias=False),
                    "adaption": Attention(
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        downsample_rate=attention_downsample_rate
                    ),
                    "w0": nn.Linear(embedding_dim, embedding_dim, bias=False),
                    "norm": nn.LayerNorm(embedding_dim),
                })
            )
        self.gate = nn.Parameter(torch.zeros(depth, 1))

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

        self.adaption_embeddings = nn.Embedding(adapter_len * depth, embedding_dim)
        self.initialize_w0()

    @torch.no_grad()
    def initialize_w0(self):
        for layer in self.layers_adapt:
            layer['w0'].weight.copy_(torch.eye(layer['w0'].weight.shape[0], layer['w0'].weight.shape[1]))

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        adaption_prompt = self.adaption_embeddings.weight.reshape(
            self.depth, self.adapter_len, self.embedding_dim
        ).unsqueeze(1)

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for i, (layer, layer_adapt) in enumerate(zip(self.layers, self.layers_adapt)):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
            # prompt_linear = layer_adapt['w_adapt'](adaption_prompt[i])
            out_adapt = layer_adapt['adaption'](
                q=keys,
                k=adaption_prompt[i],
                v=adaption_prompt[i]
            )
            # How do you combine both?
            keys = keys + out_adapt * self.gate[i]
            keys = layer_adapt['w0'](keys)
            # keys = layer_adapt['norm'](keys)  # Commented because this doesn't make it zero-init

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayTransformerBiggerAdaption(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            adapter_len: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.adapter_len = adapter_len
        self.layers = nn.ModuleList()
        self.layers_adapt = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
            self.layers_adapt.append(
                nn.ModuleDict({
                    "w_adapt": nn.Linear(embedding_dim * 2, embedding_dim, bias=False),
                    "adaption": Attention(
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        downsample_rate=attention_downsample_rate
                    ),
                    "w0": nn.Linear(embedding_dim, embedding_dim, bias=False),
                    "norm": nn.LayerNorm(embedding_dim),
                })
            )
        self.gate = nn.Parameter(torch.zeros(depth, 1))

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

        self.adaption_embeddings = nn.Embedding(adapter_len * depth, embedding_dim * 2)
        self.initialize_w0()

    @torch.no_grad()
    def initialize_w0(self):
        for layer in self.layers_adapt:
            layer['w0'].weight.copy_(torch.eye(layer['w0'].weight.shape[0], layer['w0'].weight.shape[1]))

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        adaption_prompt = self.adaption_embeddings.weight.reshape(
            self.depth, self.adapter_len, self.embedding_dim * 2
        ).unsqueeze(1)

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for i, (layer, layer_adapt) in enumerate(zip(self.layers, self.layers_adapt)):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
            prompt_linear = layer_adapt['w_adapt'](adaption_prompt[i])
            out_adapt = layer_adapt['adaption'](
                q=keys,
                k=prompt_linear,
                v=prompt_linear
            )
            # How do you combine both?
            keys = keys + out_adapt * self.gate[i]
            keys = layer_adapt['w0'](keys)
            # keys = layer_adapt['norm'](keys)  # Commented because this doesn't make it zero-init

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class ImageEncoderViTAdapter(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            adapter_len: int = 2,
            adapter_depth: int = 10,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.adapter_depth = adapter_depth
        self.adapter_len = adapter_len
        self.embedding_dim = embed_dim

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        self.blocks_adapt = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            if i >= depth - adapter_depth:
                self.blocks_adapt.append(
                    nn.ModuleDict({
                        "adaption": Attention(
                            embed_dim,
                            num_heads=num_heads,
                            downsample_rate=2
                        ),
                        "w0": nn.Linear(embed_dim, embed_dim, bias=False),
                        "norm": nn.LayerNorm(embed_dim),
                    })
                )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.gate = nn.Parameter(torch.zeros(adapter_depth, 1))
        self.adaption_embeddings = nn.Embedding(adapter_len * adapter_depth, embed_dim)
        self.initialize_w0()

    @torch.no_grad()
    def initialize_w0(self):
        for layer in self.blocks_adapt:
            layer['w0'].weight.copy_(torch.eye(layer['w0'].weight.shape[0], layer['w0'].weight.shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adaption_prompt = self.adaption_embeddings.weight.reshape(
            self.adapter_depth, self.adapter_len, self.embedding_dim
        ).unsqueeze(1)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= len(self.blocks) - self.adapter_depth:
                idx = i - len(self.blocks) + self.adapter_depth
                blk_adapter = self.blocks_adapt[idx]
                out_adapt = blk_adapter['adaption'](
                    q=x.view(x.shape[0], -1, x.shape[-1]),
                    k=adaption_prompt[idx],
                    v=adaption_prompt[idx]
                )
                out_adapt = out_adapt.view(x.shape)
                x = x + out_adapt * self.gate[idx]
                x = blk_adapter['w0'](x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
