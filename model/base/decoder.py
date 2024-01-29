import torch.nn as nn
from model.transformer.net import CPAttn


class CPBlocksDecoder(nn.Module):
    def __init__(self, up_blocks):
        super(CPBlocksDecoder, self).__init__()
        self.cp_blocks = nn.ModuleList([
            CPAttn(up_block.resnets[-1].out_channels)
            for up_block in up_blocks
        ])

    def forward(self, x):
        return [cp_block(x) for cp_block in self.cp_blocks]
