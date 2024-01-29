# Instructions to run demo file:
# Please call python demo.py -- text "prompt" --input_path "path" --output_path "path"
# Please update the prompt with a text description of the virtual room
# Please update the input path with the path to access the input video to modify
# Please update the output path with the path to save the output in

import torch
import argparse
from scripts.main import PanoGenerator
import cv2
import os
from PIL import Image
from scripts.params import params
from helper import *
from weights.download_wts import download_wts
from scripts.postprocess import generate_pano
from scripts.optical_flow import OpticalFlowVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--text', type=str,
        default='A cozy living room retreat, seamlessly fusing contemporary comfort and timeless elegance. A plush sectional sofa, adorned with neutral throw pillows, anchors the space. To the left, a minimalist fireplace emanates warmth, while to the right, expansive windows reveal a tranquil view.')

    parser.add_argument('--output_path',
                        type=str, default='output/output_video.mp4', help='output path for video / gif')
    parser.add_argument('--input_path',
                        type=str, default='output/output_video.mp4', help='output path for video / gif')

    return parser.parse_args()


def resize_and_center_crop(img, size):
    H, W, _ = img.shape
    if H == W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size * H / W)
        img = cv2.resize(img, (size, current_size))

        margin_l = (current_size - size) // 2
        margin_r = current_size - margin_l - size
        img = img[margin_l:-margin_r, :]
    else:
        current_size = int(size * W / H)
        img = cv2.resize(img, (current_size, size))
        margin_l = (current_size - size) // 2
        margin_r = current_size - margin_l - size
        img = img[:, margin_l:-margin_r]
    return img


if __name__ == "__main__":
    torch.manual_seed(0)
    params_ = params()
    args = parse_args()

    model = PanoGenerator()
    if params_.model_ckpt_path and not os.path.exists(params_.model_ckpt_path):
        download_wts()

    model.load_state_dict(torch.load(params_.model_ckpt_path, map_location='cpu')[
        'state_dict'], strict=False)
    model = model.cuda()
    img = None

    resolution = params_.resolution

    Rs = []
    Ks = []
    for i in range(8):
        degree = (45 * i) % 360
        K, R = get_K_R(90, degree, 0,
                       resolution, resolution)

        Rs.append(R)
        Ks.append(K)

    images = torch.zeros((1, 8, resolution, resolution, 3)).cuda()
    if img is not None:
        images[0, 0] = img

    prompt = [args.text] * 8
    K = torch.tensor(np.array(Ks), device='cuda')[None]
    R = torch.tensor(np.array(Rs), device='cuda')[None]

    batch = {
        'images': images,
        'prompt': prompt,
        'R': R,
        'K': K
    }
    num_images = 8
    images_pred = model.inference(batch)
    results_directory = args.text[:20]
    print('save in foldder: {}'.format(results_directory))
    os.makedirs(results_directory, exist_ok=True)
    with open(os.path.join(results_directory, 'prompt.txt'), 'w') as f:
        f.write(args.text)
    image_paths = []
    for i in range(num_images):
        image = Image.fromarray(images_pred[0, i])
        image_path = os.path.join(results_directory, '{}.png'.format(i))
        image_paths.append(image_path)
        image.save(image_path)
    generate_pano(image_paths, results_directory)

    pano_img_path = os.path.join(results_directory, 'pano.png')

    if args.input_path is not None and args.output_path is not None:
        optical_flow_visualizer = OpticalFlowVisualizer(
            args.input_path, pano_img_path, args.output_path)
        optical_flow_visualizer.visualize_optical_flow()
