""" Normalization layers and wrappers

Norm layer definitions that support fast norm and consistent channel arg order (always first arg).

Hacked together by / Copyright 2022 Ross Wightman
"""

from typing import List
import numbers
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_norm import is_fast_norm, fast_group_norm, fast_layer_norm, fast_rms_norm


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self.fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x):
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class GroupNorm1(nn.GroupNorm):
    """ Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
        self.fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm(nn.LayerNorm):
    """ LayerNorm w/ fast norm option
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self._fast_norm = is_fast_norm()  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNormAcrossScales(nn.LayerNorm):
    """ LayerNorm across scales
    """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)
        self.normalize_across_scales = True

    def forward(self, multi_x: List[torch.Tensor]) -> List[torch.Tensor]:


        B = multi_x[0].size(0)
        multi_y = []
        for x in multi_x:
            multi_y.append(x.view(B, -1, self.normalized_shape[0]))


        y = torch.concat(multi_y, dim=1)


        y = F.layer_norm(y, self.normalized_shape, self.weight, self.bias, self.eps)


        multi_y = torch.split(y, [t.size(1) for t in multi_y], dim=1)

        output_y = []
        for x, y in zip(multi_x, multi_y):
            output_y.append(y.view(B, x.size(1), x.size(2), x.size(3)))

        return output_y
#




class BatchNormAcrossScales(nn.BatchNorm1d):
    """BatchNorm across scales
    """
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            device=None,
            dtype=None
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAcrossScales, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats,
                **factory_kwargs
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAcrossScales, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

        self.normalize_across_scales = True


    def forward(self, multi_x: List[torch.Tensor]) -> List[torch.Tensor]:
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        torch._assert(multi_x[0].ndim == 4, f'expected 4D input (got {multi_x[0].ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        B = multi_x[0].size(0)
        multi_y = []
        for x in multi_x:
            multi_y.append(x.view(B, -1, self.num_features))

        y = torch.concat(multi_y, dim=1)

        y = y.permute(0, 2, 1)

        y = F.batch_norm(
            y,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        y = y.permute(0, 2, 1)

        multi_y = torch.split(y, [t.size(1) for t in multi_y], dim=1)

        output_y = []
        for x, y in zip(multi_x, multi_y):
            output_y.append(y.view(B, x.size(1), x.size(2), x.size(3)))

        return output_y



#
# class BatchNormAcrossScales(nn.Module):
#     """ BatchNorm across scales
#     """
#     def __init__(self, num_channels, eps=1e-6, affine=True):
#         super().__init__()
#         self.num_channels = num_channels
#         self.bn = nn.BatchNorm1d(num_channels, eps=eps, affine=affine)
#         self.normalize_across_scales = True
#
#     def forward(self, multi_x: List[torch.Tensor]) -> List[torch.Tensor]:
#
#
#         B = multi_x[0].size(0)
#         multi_y = []
#         for x in multi_x:
#             multi_y.append(x.view(B, -1, self.num_channels))
#
#
#         y = torch.concat(multi_y, dim=1)
#
#         y = y.permute(0, 2, 1)
#
#         self.bn(y)
#
#         y = y.permute(0, 2, 1)
#
#         multi_y = torch.split(y, [t.size(1) for t in multi_y], dim=1)
#
#         output_y = []
#         for x, y in zip(multi_x, multi_y):
#             output_y.append(y.view(B, x.size(1), x.size(2), x.size(3)))
#
#         return output_y

class BatchNorm2dCl(nn.Module):
    """ BatchNorm for channels of '2D' spatial NHWC tensors """
    def __init__(self, num_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)
        return x


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@torch.jit.script
def _layer_norm_cf(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
    x = (x - u) * torch.rsqrt(s + eps)
    x = x * weight[:, None, None] + bias[:, None, None]
    return x


def _layer_norm_cf_sqm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    u = x.mean(dim=1, keepdim=True)
    s = ((x * x).mean(dim=1, keepdim=True) - (u * u)).clamp(0)
    x = (x - u) * torch.rsqrt(s + eps)
    x = x * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    return x


class LayerNormExp2d(nn.LayerNorm):
    """ LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).

    Experimental implementation w/ manual norm for tensors non-contiguous tensors.

    This improves throughput in some scenarios (tested on Ampere GPU), esp w/ channels_last
    layout. However, benefits are not always clear and can perform worse on other GPUs.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            x = F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            x = _layer_norm_cf(x, self.weight, self.bias, self.eps)
        return x


class RmsNorm(nn.Module):
    """ RmsNorm w/ fast (apex) norm if available
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, channels, eps=1e-6, affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        normalized_shape = channels
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE fast norm fallback needs our rms norm impl, so both paths through here.
        # Since there is no built-in PyTorch impl, always use APEX RmsNorm if is installed.
        x = fast_rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return x
