import torch
import torch.nn as nn
from model.transformer.net import CPAttn
from einops import rearrange
from helper import *
from model.base.encoder import CPBlocksEncoder
from model.base.decoder import CPBlocksDecoder
from scripts.params import params


class MultiViewBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_params = params()
        self.unet_model = load_model(self.model_params.model_id, 'unet')

        self.cp_blocks_encoder = CPBlocksEncoder(self.unet_model.down_blocks)
        self.cp_blocks_mid = CPAttn(
            self.unet_model.mid_block.resnets[-1].out_channels)
        self.cp_blocks_decoder = CPBlocksDecoder(self.unet_model.up_blocks)

        self.trainable_parameters = [
            (
                list(self.cp_blocks_mid.parameters()) +
                sum([list(cp_block.parameters()) for cp_block in self.cp_blocks_encoder.cp_blocks], []) +
                sum([list(cp_block.parameters())
                    for cp_block in self.cp_blocks_decoder.cp_blocks], []),
                1.0
            )
        ]

    def forward(self, latents, timestep, prompt_embedding, meta):
        intrinsic_matrix = meta['K']
        rotation_matrix = meta['R']

        batch_size, num_views, channels, height, width = latents.shape
        img_height, img_width = height * 8, width * 8
        correspondences = get_correspondences(
            rotation_matrix, intrinsic_matrix, img_height, img_width)

        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embedding = rearrange(prompt_embedding, 'b m l c -> (b m) l c')

        timestep = timestep.reshape(-1)
        timestep_embedding = self.unet_model.time_proj(timestep)
        time_embedding = self.unet_model.time_embedding(timestep_embedding)

        hidden_states = self.unet_model.conv_in(hidden_states)

        # downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet_model.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, time_embedding)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embedding
                    ).sample()

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, time_embedding)
                    down_block_res_samples += (hidden_states,)
            if num_views > 1:
                hidden_states = self.cp_blocks_encoder.cp_blocks[i](
                    hidden_states, correspondences, img_height, img_width, rotation_matrix, intrinsic_matrix, num_views)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # mid
        hidden_states = self.unet_model.mid_block.resnets[0](
            hidden_states, time_embedding)

        if num_views > 1:
            hidden_states = self.cp_blocks_mid(
                hidden_states, correspondences, img_height, img_width, rotation_matrix, intrinsic_matrix, num_views)

        for attn, resnet in zip(self.unet_model.mid_block.attentions, self.unet_model.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embedding
            ).sample()
            hidden_states = resnet(hidden_states, time_embedding)

        h, w = hidden_states.shape[-2:]

        # upsample
        for i, upsample_block in enumerate(self.unet_model.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, time_embedding)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embedding
                    ).sample()
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, time_embedding)
            if num_views > 1:
                hidden_states = self.cp_blocks_decoder.cp_blocks[i](
                    hidden_states, correspondences, img_height, img_width, rotation_matrix, intrinsic_matrix, num_views)

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # post-process
        output = self.unet_model.conv_norm_out(hidden_states)
        output = self.unet_model.conv_act(output)
        output = self.unet_model.conv_out(output)
        output = rearrange(output, '(b m) c h w -> b m c h w', m=num_views)
        return output
