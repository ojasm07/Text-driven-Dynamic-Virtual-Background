import torch
from transformers import CLIPTextModel, CLIPTokenizer
from scripts.params import params
from helper.load_model import load_model


class Encoder():
    def __init__(self):
        self.params = params()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.params.model_id, subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.params.model_id, subfolder="text_encoder", torch_dtype=torch.float16)

        self.vae, self.scheduler, _ = load_model(
            self.params.model_id)

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        self.text_encoder = self.text_encoder.to(device)
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask)
        else:
            attention_mask = None
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device))

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        x_input = x_input.permute(0, 1, 4, 2, 3)
        x_input = x_input.reshape(-1,
                                  x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = vae.encode(x_input).latent_dist

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])

        z = z * vae.config.scaling_factor
        z = z.float()
        return z
