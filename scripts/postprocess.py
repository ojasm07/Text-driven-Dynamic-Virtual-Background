from helper import *
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def generate_pano(input_image_paths, output_directory):
    perspective_images = [cv2.imread(image_path)
                          for image_path in input_image_paths]

    inverse_eq_transform = multiple_inv_eq_transform(
        perspective_images,
        [[90, 0, 0], [90, 45, 0], [90, 90, 0], [90, 135, 0],
         [90, 180, 0], [90, 225, 0], [90, 270, 0], [90, 315, 0]]
    )

    new_pano = inverse_eq_transform.transform(2048, 4096)
    cv2.imwrite(os.path.join(output_directory, 'pano.png'),
                new_pano.astype(np.uint8)[540:-540])
