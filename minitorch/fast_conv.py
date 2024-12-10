from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Function is a decorator that uses the `njit` function from the `numba` library.
    It is used to JIT compile the decorated function for faster execution.
    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.1.
    
    for b in prange(batch_):
        for oc in prange(out_channels):
            for ow in prange(out_width):
                acc = 0.0

                # Iterate over input channels and kernel width
                for ic in range(in_channels):
                    for kw_idx in range(kw):
                        if reverse:
                            curr_w = kw - kw_idx - 1
                            curr_pos = ow - kw_idx
                        else:
                            curr_w = kw_idx
                            curr_pos = ow + kw_idx

                        # Check if within bounds of the input width
                        if 0 <= curr_pos < width:
                            in_pos = b * s1[0] + ic * s1[1] + curr_pos * s1[2]
                            w_pos = oc * s2[0] + ic * s2[1] + curr_w * s2[2]

                            acc += input[in_pos] * weight[w_pos]

                # Compute output position and store the result
                out_pos = b * out_strides[0] + oc * out_strides[1] + ow * out_strides[2]
                out[out_pos] = acc

tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    """Represents a 1D convolution operation.

    This class encapsulates the forward and backward passes for a 1D convolution operation.
    It is designed to work with tensors and supports both forward and backward propagation.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of the input and weight tensors for the 1D convolution operation.

        Args:
            ctx (Context): The context in which the operation is performed.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the gradients of the input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.

    for batch_idx in prange(batch_):
        for channel_out_idx in range(out_channels):
            for row_out_idx in range(out_height):
                for col_out_idx in range(out_width):
                    result_accumulator = 0.0

                    for channel_in_idx in range(in_channels):
                        for kernel_row_idx in range(kh):
                            for kernel_col_idx in range(kw):
                                if reverse:
                                    current_row = row_out_idx - kh + 1 + kernel_row_idx
                                    current_col = col_out_idx - kw + 1 + kernel_col_idx
                                else:
                                    current_row = row_out_idx + kernel_row_idx
                                    current_col = col_out_idx + kernel_col_idx

                                if 0 <= current_row < height and 0 <= current_col < width:
                                    input_position = (
                                        batch_idx * s10
                                        + channel_in_idx * s11
                                        + current_row * s12
                                        + current_col * s13
                                    )
                                    weight_position = (
                                        channel_out_idx * s20
                                        + channel_in_idx * s21
                                        + kernel_row_idx * s22
                                        + kernel_col_idx * s23
                                    )

                                    result_accumulator += (
                                        input[input_position] * weight[weight_position]
                                    )

                    output_position = (
                        batch_idx * out_strides[0]
                        + channel_out_idx * out_strides[1]
                        + row_out_idx * out_strides[2]
                        + col_out_idx * out_strides[3]
                    )
                    out[output_position] = result_accumulator



    
tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    """Class represents a 2D convolution operation. It is a subclass of the Function class from the autodiff module.
    It is designed to compute a 2D convolution between an input tensor and a weight tensor, producing an output tensor.
    The forward method computes the convolution, and the backward method computes the gradients of the loss with respect to the input and weights.
    """

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of the loss with respect to the input and weights for a 2D convolution operation.

        Args:
        ----
            ctx : Context
                The context in which the forward operation was performed.
            grad_output : Tensor
                The gradient of the loss with respect to the output of the convolution operation.

        Returns:
        -------
            Tuple[Tensor, Tensor]
                A tuple containing the gradients of the loss with respect to the input and weights.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
