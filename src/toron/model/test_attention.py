import torch
from toron.model.config import ModelConfig
from toron.model.attention import MultiHeadAttention


def test_attention():
    config = ModelConfig()
    model = MultiHeadAttention(config)
    model.eval()
    x = torch.randn(2, 8, config.hidden_size)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0]
    ])
    with torch.inference_mode():
        output = model(x, mask)
        modified = x.clone()
        modified[1, 5:] = 100
        second_output = model(modified, mask)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output[1, :5],
        second_output[1, :5]
    )
    print("Output shape:", output.shape)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))