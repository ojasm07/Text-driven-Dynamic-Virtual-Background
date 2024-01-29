import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from helper import *
from model.base.model import MultiViewBaseModel
from tqdm import tqdm


class PanoGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.params = params()
        self.lr = self.params.lr
        self.max_epochs = getattr(self.params, 'max_epochs', 0)
        self.diff_timestep = self.params.diff_timestep
        self.guidance_scale = self.params.guidance_scale

        self.vae, self.scheduler, _ = load_model(
            self.params.model_id)
        self.mv_base_model = MultiViewBaseModel()
        self.trainable_params = self.mv_base_model.trainable_parameters

        self.encoder = Encoder()
        self.decoder = Decoder()

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        meta = {
            'K': batch['K'],
            'R': batch['R']
        }

        device = batch['images'].device
        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encoder.encode_text(
                prompt, device)[0])
        latents = self.encoder.encode_image(
            batch['images'], self.vae)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                          (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=1)

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])
        denoise = self.mv_base_model(
            noise_z, t, prompt_embds, meta)
        target = noise

        # eps mode
        loss = F.mse_loss(denoise, target)
        self.log('train_loss', loss)
        return loss

    @torch.no_grad()
    def generate_guided_noise_prediction(self, latents_high_res, _timestep, prompt_embd, batch, model):

        latents = torch.cat([latents_high_res]*2)
        _timestep = torch.cat([_timestep]*2)

        R = torch.cat([batch['R']]*2)
        K = torch.cat([batch['K']]*2)

        meta = {
            'K': K,
            'R': R,
        }

        _prompt_embd = prompt_embd

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        images = ((batch['images']/2+0.5)
                  * 255).cpu().numpy().astype(np.uint8)

        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch['prompt'], batch_idx)

    @torch.no_grad()
    def inference(self, batch):
        images = batch['images']
        bs, m, h, w, _ = images.shape
        device = images.device

        latents = torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encoder.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encoder.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(tqdm(timesteps, desc="Processing timesteps", unit="step")):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.generate_guided_noise_prediction(
                latents, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        images_pred = self.decoder.decode(
            latents, self.vae)

        return images_pred

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images_pred = self.inference(batch)

        images = ((batch['images']/2+0.5)
                  * 255).cpu().numpy().astype(np.uint8)

        scene_id = batch['image_paths'][0].split('/')[2]
        image_id = batch['image_paths'][0].split(
            '/')[-1].split('.')[0].split('_')[0]

        output_dir = batch['resume_dir'][0] if 'resume_dir' in batch else os.path.join(
            self.logger.log_dir, 'images')
        output_dir = os.path.join(
            output_dir, "{}_{}".format(scene_id, image_id))

        os.makedirs(output_dir, exist_ok=True)
        for i in range(images.shape[1]):
            path = os.path.join(output_dir, f'{i}.png')
            im = Image.fromarray(images_pred[0, i])
            im.save(path)
            im = Image.fromarray(images[0, i])
            path = os.path.join(output_dir, f'{i}_natural.png')
            im.save(path)
        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            for p in batch['prompt']:
                f.write(p[0]+'\n')

    @torch.no_grad()
    def save_image(self, images_pred, images, prompt, batch_idx):

        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
            for p in prompt:
                f.write(p[0]+'\n')
        if images_pred is not None:
            for m_i in range(images_pred.shape[1]):
                im = Image.fromarray(images_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))
                if m_i < images.shape[1]:
                    im = Image.fromarray(
                        images[0, m_i])
                    im.save(os.path.join(
                        img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))
