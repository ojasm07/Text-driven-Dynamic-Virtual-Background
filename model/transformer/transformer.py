import torch.nn as nn
from model.transformer.grad_checkpoint import checkpoint
from model.transformer.attention_block import MultiHeadAttention
from model.transformer.forward import Forward


class Transformer(nn.Module):
    def __init__(self, dim, n_heads, d_heads, dropout=0., context_dim=None, ):
        super().__init__()

        self.attn = MultiHeadAttention(dim, context_dim, n_heads,
                                       d_heads)
        self.feed_forward = Forward(dim, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def _forward(self, embeddings, context=None, query_positional_encoding=None):
        context = query_positional_encoding = embeddings
        query = self.layer_norm1(query_positional_encoding)
        context = self.layer_norm1(context)

        embeddings = embeddings + self.attn(query, context)
        embeddings = embeddings + \
            self.feed_forward(self.layer_norm2(embeddings))

        return embeddings

    def forward(self, embeddings, context=None, query_positional_encoding=None):
        return checkpoint(self._forward, (embeddings, context, query_positional_encoding), self.parameters(), self.checkpoint)
