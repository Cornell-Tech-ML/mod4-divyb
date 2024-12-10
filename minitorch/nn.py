"""Neural network operations and utilities for the MiniTorch library.
Implements core neural network operations including pooling, softmax, dropout,
and various tensor manipulation functions used in deep learning.
"""

from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See
#           https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    pooled_height = height // kh
    pooled_width = width // kw

    reshaped_tensor = input.contiguous().view(
        batch, channel, pooled_height, kh, pooled_width, kw
    )

    permuted_tensor = reshaped_tensor.permute(0, 1, 2, 4, 3, 5).contiguous()
    final_output = permuted_tensor.view(
        batch, channel, pooled_height, pooled_width, kh * kw
    )

    return final_output, pooled_height, pooled_width


# TODO: Implement for Task 4.3.


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies a 2D average pooling over an input signal composed of several input planes.
    
    Args:
        input: Input tensor of shape (batch, channel, height, width)
        kernel: Tuple of (kernel_height, kernel_width) for pooling
        
    Returns:
        Pooled output tensor
        
    """
    reshaped_tensor, pooled_height, pooled_width = tile(input, kernel)
    return (
        reshaped_tensor.mean(4)
        .contiguous()
        .view(
            reshaped_tensor.shape[0],
            reshaped_tensor.shape[1],
            pooled_height,
            pooled_width,
        )
    )


fastmax = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Returns a tensor indicating the positions of the maximum values in the input tensor along the specified dimension.
    
    Args:
        input: Input tensor to find maximum values in
        dim: Dimension along which to find maximum values
        
    Returns:
        Binary tensor with ones at maximum value positions
        
    """
    max_tensor = fastmax(input, dim)
    return max_tensor == input


class Max(Function):
    """Max Function"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the maximum value along a specified dimension."""
        ctx.save_for_backward(a, int(dim.item()))
        return fastmax(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_out: Tensor) -> Tuple[Tensor, float]:
        """Backward"""
        saved_input, saved_dim = ctx.saved_values
        return grad_out * argmax(saved_input, saved_dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Applies the Max function to the input tensor along the specified dimension.
    
    Args:
        input: Input tensor to find maximum values in
        dim: Dimension along which to find maximum values
        
    Returns:
        Tensor containing maximum values along specified dimension

    """
    return Max.apply(input, input._ensure_tensor(dim))

def softmax(input: Tensor, dim: int) -> Tensor:
    """Softmax"""
    exp_tensor = input.exp()
    exp_sum = exp_tensor.sum(dim)
    return exp_tensor / exp_sum


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Logsofmax"""
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Maxpool2d"""
    batch_size, num_channels, img_height, img_width = input.shape
    tiled_tensor, pooled_height, pooled_width = tile(input, kernel)
    return (
        max(tiled_tensor, dim=4)
        .contiguous()
        .view(batch_size, num_channels, pooled_height, pooled_width)
    )


def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:    
    """Apply dropout to the input tensor.
    
    Args:
        input: Input tensor to apply dropout to
        prob: Probability of dropping a unit (setting it to 0)
        ignore: If True, no dropout is applied
        
    Returns:
        Tensor with dropout applied

    """
    if ignore or prob == 0.0:
        return input

    if prob == 1.0:
        return input.zeros()

    dropout_mask = rand(input.shape) > prob
    return input * dropout_mask
