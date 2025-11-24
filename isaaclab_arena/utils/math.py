import torch


def normalize_value(value: torch.Tensor, min_value: float, max_value: float):
    return (value - min_value) / (max_value - min_value)


def unnormalize_value(value: float, min_value: float, max_value: float):
    return min_value + (max_value - min_value) * value
