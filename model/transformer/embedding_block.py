import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, input_channels, num_frequencies):
        super(Embedding, self).__init__()
        self.input_channels = input_channels
        self.num_frequencies = num_frequencies

        if num_frequencies <= 80:
            base = 2
        else:
            base = 5000**(1 / (num_frequencies / 2.5))

        frequency_bands = base**torch.linspace(
            0, num_frequencies - 1, num_frequencies)[None, None]
        self.register_buffer('frequency_bands', frequency_bands)

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, 2, 1)
        encodings = x * self.frequency_bands
        sin_encodings = torch.sin(encodings)
        cos_encodings = torch.cos(encodings)
        positional_encodings = torch.cat([sin_encodings, cos_encodings], dim=1)

        return positional_encodings.reshape(*shape, -1)
