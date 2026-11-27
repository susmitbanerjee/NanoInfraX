import math
import torch
from torch import nn
from toron.model.config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def split_heads(self, x):
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_attention_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attention_mask=None):
        batch, seq, _ = x.shape
        q=self.split_heads(self.query(x))
        k=self.split_heads(self.key(x))
        v=self.split_heads(self.value(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch, seq, self.hidden_size)
        output=self.output(context)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        return output