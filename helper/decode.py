import torch


class Decoder():
    def __init__(self):
        pass

    @torch.no_grad()
    def decode(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image
