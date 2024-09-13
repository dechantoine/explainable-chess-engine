import torch


def get_model(model_name: str) -> torch.nn.Module:
    if model_name == "multi_input_feedforward":
        from .multi_input_feedforward import MultiInputFF
        return MultiInputFF()
    elif model_name == "multi_input_conv":
        from .multi_input_conv import MultiInputConv
        return MultiInputConv()
    elif model_name == "multi_input_attention":
        from .multi_input_attention import AttentionModel
        return AttentionModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
