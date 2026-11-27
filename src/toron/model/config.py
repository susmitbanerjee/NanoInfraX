from dataclasses import dataclass
@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 192
    max_sequence_length: int = 256
    num_attention_heads: int = 4
    num_layers: int = 6
    intermediate_size: int = 768
    dropout: float = 0.1
    pad_token_id: int = 0