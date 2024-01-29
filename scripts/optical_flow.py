import cv2
import numpy as np
import torch
from PIL import Image
import PIL
from torchvision import transforms
import imageio


class OpticalFlowVisualizer:
    def __init__(self, video_path, panoramic_background_path, output_video_path, window_size=400):
        self.deeplab_model = self.load_model()

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(
            output_video_path, fourcc, self.fps, (self.width, self.height))

        self.panoramic_background_path = panoramic_background_path
        self.window_size = window_size

        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.old_frame = None
        self.old_gray = None
        self.p0 = None
        self.frame_count = 0

    def load_model(self):
        model = torch.hub.load('pytorch/vision:v0.6.0',
                               'deeplabv3_resnet101', pretrained=True)
        model.eval()
        return model

    def make_transparent_foreground(self, pic, mask):
        b, g, r = cv2.split(np.array(pic).astype('uint8'))
        a = np.ones(mask.shape, dtype='uint8') * 255
        alpha_im = cv2.merge([b, g, r, a], 4)
        bg = np.zeros(alpha_im.shape)
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground

    def remove_background(self, frame):
        input_image = Image.fromarray(frame)
        input_image_np = np.array(input_image)

        input_image_rgb = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image_rgb)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.deeplab_model.to('cuda')

        with torch.no_grad():
            output = self.deeplab_model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask, 255, background).astype(np.uint8)

        bin_mask = cv2.threshold(bin_mask, 100, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((7, 7), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        foreground = self.make_transparent_foreground(
            input_image_rgb, bin_mask)

        return foreground, bin_mask

    def custom_background(self, foreground, optical_flow_direction):
        final_foreground = Image.fromarray(foreground)
        background = Image.open(self.panoramic_background_path)
        scale_factor = 1
        scaled_optical_flow = scale_factor * optical_flow_direction

        shifted_background = background.copy()
        shifted_background = shifted_background.transform(shifted_background.size, Image.AFFINE,
                                                          (1, 0, -scaled_optical_flow[0], 0, 1, -scaled_optical_flow[1]))

        window_position = (
            int(shifted_background.size[0] / 2), int(shifted_background.size[1] / 2))
        window_position = tuple(np.subtract(
            window_position, np.multiply(scaled_optical_flow, self.window_size)))

        window_position = (
            max(0, min(
                window_position[0], shifted_background.size[0] - self.window_size)),
            max(0, min(
                window_position[1], shifted_background.size[1] - self.window_size))
        )

        self.window_position = window_position

        box = (
            int(window_position[0]),
            int(window_position[1]),
            int(window_position[0] + self.window_size),
            int(window_position[1] + self.window_size),
        )
        cropped_background = shifted_background.crop(box)

        resized_foreground = final_foreground.resize(
            cropped_background.size, PIL.Image.Resampling.LANCZOS)

        final_image = cropped_background.copy()
        paste_box = (0, final_image.size[1] - resized_foreground.size[1],
                     final_image.size[0], final_image.size[1])
        final_image.paste(resized_foreground, paste_box,
                          mask=resized_foreground)

        return final_image

    def linear_interpolation(self, start_position, flow_direction, alpha):
        interpolated_position = (
            int(start_position[0] + alpha * flow_direction[0]),
            int(start_position[1] + alpha * flow_direction[1])
        )
        return interpolated_position

    def custom_interpolated_background(self, foreground, window_position):
        final_foreground = Image.fromarray(foreground)
        background = Image.open(self.panoramic_background_path)
        background = background.copy()

        window_position = (
            max(0, min(
                window_position[0], background.size[0] - self.window_size)),
            max(0, min(
                window_position[1], background.size[1] - self.window_size))
        )

        box = (
            int(window_position[0]),
            int(window_position[1]),
            int(window_position[0] + self.window_size),
            int(window_position[1] + self.window_size),
        )
        cropped_background = background.crop(box)

        resized_foreground = final_foreground.resize(
            cropped_background.size, PIL.Image.Resampling.LANCZOS)

        final_image = cropped_background.copy()
        paste_box = (0, final_image.size[1] - resized_foreground.size[1],
                     final_image.size[0], final_image.size[1])
        final_image.paste(resized_foreground, paste_box,
                          mask=resized_foreground)

        return final_image

    def visualize_optical_flow(self):
        frame_count = 0
        skip_frames = 10
        frames_for_gif = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % skip_frames != 0:
                alpha = (frame_count % skip_frames) * skip_frames
                window_position = self.linear_interpolation(
                    self.window_position, self.flow_direction, alpha * skip_frames)

                result_frame, bin_mask = self.remove_background(frame)

                final_background = self.custom_interpolated_background(
                    result_frame, window_position)
                final_background_np = np.array(final_background)

            else:
                if self.old_frame is None:
                    self.old_frame = frame
                    self.old_gray = cv2.cvtColor(
                        self.old_frame, cv2.COLOR_BGR2GRAY)
                    self.p0 = cv2.goodFeaturesToTrack(
                        self.old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                    continue

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    self.old_gray, frame_gray, self.p0, None, **self.lk_params)

                good_new = p1[st == 1]
                good_old = self.p0[st == 1]

                if len(good_new) > 0:
                    flow_direction = np.mean(good_new - good_old, axis=0)

                self.flow_direction = flow_direction
                result_frame, bin_mask = self.remove_background(frame)

                final_background = self.custom_background(
                    result_frame, flow_direction)
                final_background_np = np.array(final_background)

            cv2.imshow('Optical Flow', final_background_np)

            self.out.write(final_background_np)

            frames_for_gif.append(final_background_np)

            k = cv2.waitKey(1)
            print(k)
            if k == 27:
                break

            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)

            frame_count += 1

        self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()
        gif_path = 'output_animation2.gif'
        imageio.mimsave(gif_path, frames_for_gif, duration=1 / self.fps)
        print(f'GIF saved at: {gif_path}')


if __name__ == "__main__":
    video_path = r'Test_videos/Akash3.mp4'
    panoramic_background_path = r'Test_videos/Input.jpg'

    output_video_path = 'output_video.mp4'

    optical_flow_visualizer = OpticalFlowVisualizer(
        video_path, panoramic_background_path, output_video_path)
    optical_flow_visualizer.visualize_optical_flow()
