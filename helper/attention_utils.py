import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


def compute_key_values_and_rel_pos(key_value_tensor: torch.Tensor,
                                   input_coords: torch.Tensor,
                                   homography_matrix: torch.Tensor,
                                   original_height: int,
                                   original_width: int,
                                   homography_height: int,
                                   query_height: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size, channels, height, width = key_value_tensor.shape
    query_scale = original_height // query_height
    key_scale = homography_height // height

    normalized_input_coords = input_coords[
        :, query_scale // 2::query_scale, query_scale // 2::query_scale
    ] / key_scale - 0.5

    computed_key_values = []
    computed_projected_coords = []
    kernel_size = 3

    for i in range(0 - kernel_size // 2, 1 + kernel_size // 2):
        for j in range(0 - kernel_size // 2, 1 + kernel_size // 2):
            modified_coords = normalized_input_coords.clone()
            modified_coords[..., 0] = modified_coords[..., 0] + i
            modified_coords[..., 1] = modified_coords[..., 1] + j
            rescaled_coords = (modified_coords + 0.5) * key_scale

            computed_projected_coords.append(rescaled_coords)

            modified_coords[..., 0] = modified_coords[...,
                                                      0] / (width - 1) * 2 - 1
            modified_coords[..., 1] = modified_coords[...,
                                                      1] / (height - 1) * 2 - 1
            interpolated_key_value = F.grid_sample(
                key_value_tensor, modified_coords, align_corners=True
            )
            computed_key_values.append(interpolated_key_value)

    computed_projected_coords = torch.stack(
        computed_projected_coords, dim=1)
    valid_coords_mask = (
        (computed_projected_coords[..., 0] > 0)
        * (computed_projected_coords[..., 0] < original_width)
        * (computed_projected_coords[..., 1] > 0)
        * (computed_projected_coords[..., 1] < original_height)
    )

    concatenated_projected_coords = torch.cat(
        [computed_projected_coords, torch.ones(
            *computed_projected_coords.shape[:-1], 1, device=computed_projected_coords.device)
         ],
        dim=-1
    )
    concatenated_projected_coords = rearrange(
        concatenated_projected_coords, 'b n h w c -> b c (n h w)'
    )
    concatenated_projected_coords = homography_matrix @ concatenated_projected_coords

    concatenated_projected_coords = rearrange(
        concatenated_projected_coords, 'b c (n h w) -> b n h w c', h=height, w=width
    )
    concatenated_projected_coords = (
        concatenated_projected_coords[..., :2]
        / concatenated_projected_coords[..., 2:]
    )

    x = np.arange(original_width)
    y = np.arange(original_height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    reference_coords = np.concatenate(
        [x[..., None], y[..., None], z[..., None]], axis=-1
    ).astype(np.float32)

    reference_coords = reference_coords[
        :, :, :2
    ]
    reference_coords = reference_coords[
        query_scale // 2::query_scale, query_scale // 2::query_scale
    ]
    reference_coords = torch.tensor(
        reference_coords, device=key_value_tensor.device
    ).float()[None, None]

    relative_coords = (
        concatenated_projected_coords - reference_coords) / query_scale

    computed_key_values = torch.stack(computed_key_values, dim=1)

    return computed_key_values, relative_coords, valid_coords_mask


def retrieve_query_values(query_tensor: torch.Tensor,
                          key_value_tensor: torch.Tensor,
                          input_coords_tensor: torch.Tensor,
                          homography_matrix_tensor: torch.Tensor,
                          img_height_left: int,
                          img_width_left: int,
                          img_height_right: int = None,
                          img_width_right: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if img_height_right is None:
        img_height_right = img_height_left
        img_width_right = img_width_left

    batch_size = query_tensor.shape[0]
    num_key_values = key_value_tensor.shape[1]

    retrieved_key_values = []
    masks = []
    relative_coords = []

    for i in range(num_key_values):
        _, _, query_height, query_width = query_tensor.shape
        computed_key_values, computed_relative_coords, computed_mask = compute_key_values_and_rel_pos(
            key_value_tensor[:, i], input_coords_tensor[:,
                                                        i], homography_matrix_tensor[:, i],
            img_height_left, img_width_left, img_width_right, query_height
        )

        retrieved_key_values.append(computed_key_values)
        relative_coords.append(computed_relative_coords)
        masks.append(computed_mask)

    key_value_tensor_retrieved = torch.cat(retrieved_key_values, dim=1)
    xy_coords = torch.cat(relative_coords, dim=1)
    mask_tensor = torch.cat(masks, dim=1)

    return query_tensor, key_value_tensor_retrieved, xy_coords, mask_tensor
