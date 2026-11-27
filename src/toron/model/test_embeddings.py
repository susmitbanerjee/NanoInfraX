import torch
from pathlib import Path
from tokenizers import Tokenizer
from toron.model.config import ModelConfig
from toron.model.embeddings import Embeddings

def test_embeddings():
    tokenizer_path = (Path(__file__).resolve().parents[4] / "models" / "tokenizer" / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size(), pad_token_id=tokenizer.token_to_id("[PAD]"))
    model = Embeddings(config)
    model.eval()
    encoded = tokenizer.encode("Connection refused")
    input_ids = torch.tensor([encoded.ids], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids)
    print("input shape:", input_ids.shape)
    print("output shape:", output.shape)
    parameters = sum(p.numel() for p in model.parameters())
    print("Parameters:", parameters)
    assert output.shape == (1, len(encoded.ids), config.hidden_size)