import cv2
import numpy as np
import torch
from typing import Tuple, List, Union


def get_correspondences(R: torch.Tensor, K: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    num_cameras = R.shape[1]

    correspondences = torch.zeros(
        (R.shape[0], num_cameras, num_cameras, image_height, image_width, 2), device=R.device)

    for i in range(num_cameras):
        for j in range(num_cameras):
            rotation_right = R[:, j:j+1]
            intrinsic_right = K[:, j:j+1]
            l = rotation_right.shape[1]

            rotation_left = R[:, i:i+1].repeat(1, l, 1, 1)
            intrinsic_left = K[:, i:i+1].repeat(1, l, 1, 1)

            rotation_left = rotation_left.reshape(-1, 3, 3)
            rotation_right = rotation_right.reshape(-1, 3, 3)
            intrinsic_left = intrinsic_left.reshape(-1, 3, 3)
            intrinsic_right = intrinsic_right.reshape(-1, 3, 3)

            homogeneous_left = (intrinsic_right @ torch.inverse(rotation_right) @
                                rotation_left @ torch.inverse(intrinsic_left))

            x = np.arange(image_width)
            y = np.arange(image_height)
            x, y = np.meshgrid(x, y)
            z = np.ones_like(x)
            xyz = np.concatenate(
                [x[..., None], y[..., None], z[..., None]], axis=-1).astype(np.float32)

            xyz_left = torch.tensor(xyz, device=R.device)
            xyz_left = (
                xyz_left.reshape(-1, 3).T)[None].repeat(homogeneous_left.shape[0], 1, 1)

            xyz_left = homogeneous_left @ xyz_left

            xy_left = (xyz_left[:, :2]/xyz_left[:, 2:]).permute(0,
                                                                2, 1).reshape(-1, l, image_height, image_width, 2)

            correspondences[:, i, j] = xy_left[:, 0]

    return correspondences


def get_K_R(FOV: float, THETA: float, PHI: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


class equirectangular_transform:
    def __init__(self, image_path: str, text_to_light: bool = False):
        if isinstance(image_path, str):
            self._image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            self._image = image_path
        if text_to_light:
            self._image = np.roll(self._image, -60, axis=0)

        [self._image_height, self._image_width, _] = self._image.shape

    def transform(self, FOV: float, theta: float, phi: float, output_height: int, output_width: int) -> np.ndarray:
        equirectangular_height = self._image_height
        equirectangular_width = self._image_width
        equirectangular_center_x = (equirectangular_width - 1) / 2.0
        equirectangular_center_y = (equirectangular_height - 1) / 2.0

        horizontal_field_of_view = FOV
        vertical_field_of_view = float(
            output_height) / output_width * horizontal_field_of_view

        horizontal_len = np.tan(np.radians(horizontal_field_of_view / 2.0))
        vertical_len = np.tan(np.radians(vertical_field_of_view / 2.0))

        x_map = np.ones([output_height, output_width], np.float32)
        y_map = np.tile(np.linspace(-horizontal_len,
                        horizontal_len, output_width), [output_height, 1])
        z_map = -np.tile(np.linspace(-vertical_len, vertical_len,
                         output_height), [output_width, 1]).T

        distance = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map, y_map, z_map), axis=2) / \
            np.repeat(distance[:, :, np.newaxis], 3, axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [rotation_1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
        [rotation_2, _] = cv2.Rodrigues(
            np.dot(rotation_1, y_axis) * np.radians(-phi))

        xyz = xyz.reshape([output_height * output_width, 3]).T
        xyz = np.dot(rotation_1, xyz)
        xyz = np.dot(rotation_2, xyz).T
        latitude = np.arcsin(xyz[:, 2])
        longitude = np.arctan2(xyz[:, 1], xyz[:, 0])

        longitude = longitude.reshape(
            [output_height, output_width]) / np.pi * 180
        latitude = - \
            latitude.reshape([output_height, output_width]) / np.pi * 180

        longitude = longitude / 180 * equirectangular_center_x + equirectangular_center_x
        latitude = latitude / 90 * equirectangular_center_y + equirectangular_center_y

        perspective = cv2.remap(self._image, longitude.astype(np.float32), latitude.astype(
            np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return perspective


class inverse_equirectangular_transform:
    def __init__(self, image_path: str, field_of_view: float, theta: float, phi: float):
        if isinstance(image_path, str):
            self._image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        else:
            self._image = image_path
        [self._image_height, self._image_width, _] = self._image.shape
        self.horizontal_field_of_view = field_of_view
        self.theta = theta
        self.phi = phi
        self.vertical_field_of_view = float(
            self._image_height) / self._image_width * field_of_view

        self.horizontal_len = np.tan(np.radians(
            self.horizontal_field_of_view / 2.0))
        self.vertical_len = np.tan(np.radians(
            self.vertical_field_of_view / 2.0))

    def transform(self, output_height: int, output_width: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = np.meshgrid(np.linspace(-180, 180, output_width),
                           np.linspace(90, -90, output_height))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map), axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [rotation_1, _] = cv2.Rodrigues(z_axis * np.radians(self.theta))
        [rotation_2, _] = cv2.Rodrigues(
            np.dot(rotation_1, y_axis) * np.radians(-self.phi))

        rotation_1_inv = np.linalg.inv(rotation_1)
        rotation_2_inv = np.linalg.inv(rotation_2)

        xyz = xyz.reshape([output_height * output_width, 3]).T
        xyz = np.dot(rotation_2_inv, xyz)
        xyz = np.dot(rotation_1_inv, xyz).T

        xyz = xyz.reshape([output_height, output_width, 3])
        inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

        xyz[:, :] = xyz[:, :] / \
            np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

        longitude_map = np.where((-self.horizontal_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.horizontal_len) &
                                 (-self.vertical_len <
                                  xyz[:, :, 2]) & (xyz[:, :, 2] < self.vertical_len),
                                 (xyz[:, :, 1]+self.horizontal_len)/2/self.horizontal_len*self._image_width, 0)
        latitude_map = np.where((-self.horizontal_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.horizontal_len) &
                                (-self.vertical_len <
                                 xyz[:, :, 2]) & (xyz[:, :, 2] < self.vertical_len),
                                (-xyz[:, :, 2]+self.vertical_len)/2/self.vertical_len*self._image_height, 0)
        mask = np.where((-self.horizontal_len < xyz[:, :, 1]) & (xyz[:, :, 1] < self.horizontal_len) &
                        (-self.vertical_len < xyz[:, :, 2]) & (xyz[:, :, 2] < self.vertical_len), 1, 0)

        perspective = cv2.remap(self._image, longitude_map.astype(np.float32), latitude_map.astype(
            np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        perspective = perspective * mask

        return perspective, mask


class multiple_inv_eq_transform:
    def __init__(self, img_array: List[Union[str, np.ndarray]], F_T_P_array: List[List[float]]):

        assert len(img_array) == len(F_T_P_array)

        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

    def transform(self, height: int, width: int) -> np.ndarray:
        merge_image = np.zeros((height, width, 3))
        merge_mask = np.zeros((height, width, 3))
        for img_dir, [F, T, P] in zip(self.img_array, self.F_T_P_array):
            per = inverse_equirectangular_transform(img_dir, F, T, P)
            img, mask = per.transform(height, width)
            mask = mask.astype(np.float32)
            img = img.astype(np.float32)
            weight_mask = np.zeros((img_dir.shape[0], img_dir.shape[1], 3))
            w = img_dir.shape[1]
            weight_mask[:, 0:w//2, :] = np.linspace(0, 1, w//2)[..., None]
            weight_mask[:, w//2:, :] = np.linspace(1, 0, w//2)[..., None]
            weight_mask = inverse_equirectangular_transform(
                weight_mask, F, T, P)
            weight_mask, _ = weight_mask.transform(height, width)
            blur = cv2.blur(mask, (5, 5))
            blur = blur * mask
            mask = (blur == 1) * blur + (blur != 1) * blur * 0.05
            merge_image += img * weight_mask
            merge_mask += weight_mask
        merge_image[merge_mask == 0] = 255.
        merge_mask = np.where(merge_mask == 0, 1, merge_mask)
        merge_image = (np.divide(merge_image, merge_mask))

        return merge_image
