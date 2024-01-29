import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dimension, context_dimension, num_heads, head_dimension):
        super().__init__()
        inner_dimension = head_dimension * num_heads
        context_dimension = context_dimension

        self.scale = head_dimension ** -0.5
        self.num_heads = num_heads

        self.query_projection = nn.Linear(
            query_dimension, inner_dimension, bias=False)
        self.key_projection = nn.Linear(
            context_dimension, inner_dimension, bias=False)
        self.value_projection = nn.Linear(
            context_dimension, inner_dimension, bias=False)

        self.output_projection = nn.Linear(inner_dimension, query_dimension)
        nn.init.zeros_(self.output_projection.weight.data)
        nn.init.zeros_(self.output_projection.bias.data)

    def forward(self, queries, context=None, mask=None):
        h = self.num_heads

        query = self.query_projection(queries)
        key = self.key_projection(context)
        value = self.value_projection(context)

        query, key, value = [tensor.view(tensor.size(0), -1, h, tensor.size(-1)//h).transpose(1, 2)
                             for tensor in (query, key, value)]

        similarity = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        similarity = similarity.softmax(dim=-1)

        output = torch.matmul(similarity, value)
        output = output.transpose(1, 2).contiguous().view(
            output.size(0), -1, h * value.size(-1))

        return self.output_projection(output)
