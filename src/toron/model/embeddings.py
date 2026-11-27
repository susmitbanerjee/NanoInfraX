import torch
from torch import nn
from toron.model.config import ModelConfig

class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.positional_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        tokens = self.token_embeddings(input_ids)
        positions = self.positional_embeddings(position_ids)
        x = tokens + positions
        return self.dropout(self.layer_norm(x))
