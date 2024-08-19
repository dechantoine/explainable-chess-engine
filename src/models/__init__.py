import torch


def get_model(model_name: str) -> torch.nn.Module:
    if model_name == "simple_feed_forward":
        from .simple_feed_forward import SimpleFF
        return SimpleFF()
    elif model_name == "multi_input_feedforward":
        from .multi_input_feedforward import MultiInputFF
        return MultiInputFF()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
