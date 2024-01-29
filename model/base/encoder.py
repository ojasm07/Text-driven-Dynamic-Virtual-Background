from model.transformer.net import CPAttn
import torch.nn as nn


class CPBlocksEncoder(nn.Module):
    def __init__(self, down_blocks):
        super(CPBlocksEncoder, self).__init__()
        self.cp_blocks = nn.ModuleList([
            CPAttn(down_block.resnets[-1].out_channels)
            for down_block in down_blocks
        ])

    def forward(self, x):
        return [cp_block(x) for cp_block in self.cp_blocks]
