import torch
import torch.nn as nn
from einops import rearrange
from model.transformer.transformer import Transformer
from model.transformer.embedding_block import Embedding
from helper import *


class CPAttn(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.transformer = Transformer(
            embedding_dim, embedding_dim // 32, 32, context_dim=embedding_dim)
        self.positional_embedding = Embedding(2, embedding_dim // 4)

    def forward(self, x, correspondences, image_height, image_width, R, K, num_views):
        batch_size, channels, height, width = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=num_views)
        outputs = []

        for i in range(num_views):
            indices = [(i - 1 + num_views) % num_views, (i + 1) % num_views]

            xy_left = correspondences[:, i, indices]
            xy_right = correspondences[:, indices, i]

            x_left = x[:, i]
            x_right = x[:, indices]

            R_right = R[:, indices]
            K_right = K[:, indices]

            num_indices = R_right.shape[1]

            R_left = R[:, i:i+1].repeat(1, num_indices, 1, 1)
            K_left = K[:, i:i+1].repeat(1, num_indices, 1, 1)

            R_left = R_left.reshape(-1, 3, 3)
            R_right = R_right.reshape(-1, 3, 3)
            K_left = K_left.reshape(-1, 3, 3)
            K_right = K_right.reshape(-1, 3, 3)

            homography_right = (K_left @ torch.inverse(R_left)
                                @ R_right @ torch.inverse(K_right))

            homography_right = rearrange(
                homography_right, '(b l) h w -> b l h w', b=xy_right.shape[0])
            query, key_value, key_value_xy, mask = retrieve_query_values(
                x_left, x_right, xy_left, homography_right, image_height, image_width)

            key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')
            key_value_positional_encoding = self.positional_embedding(
                key_value_xy)

            key_value = rearrange(key_value, 'b l c h w-> (b h w) l c')
            mask = rearrange(mask, 'b l h w -> (b h w) l')

            key_value = (
                key_value + key_value_positional_encoding) * mask[..., None]

            query = rearrange(query, 'b c h w->(b h w) c')[:, None]
            query_positional_encoding = self.positional_embedding(torch.zeros(
                query.shape[0], 1, 2, device=query.device))

            out = self.transformer(
                query, key_value, query_pe=query_positional_encoding)

            out = rearrange(
                out[:, 0], '(b h w) c -> b c h w', h=height, w=width)
            outputs.append(out)
        output = torch.stack(outputs, dim=1)

        output = rearrange(output, 'b m c h w -> (b m) c h w')

        return output
