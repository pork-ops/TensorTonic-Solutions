import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = torch.sum(weights * inputs) + bias
    y = torch.tanh(a)
    delta = upstream_gradient * (1 - y.square())
    input_gradients = delta * weights
    weight_gradients = delta * inputs
    bias_gradient = delta
    return (y, input_gradients, weight_gradients, bias_gradient)