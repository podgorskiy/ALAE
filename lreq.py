# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np


class Bool:
    def __init__(self):
        self.value = False

    def __bool__(self):
        return self.value
    __nonzero__ = __bool__

    def set(self, value):
        self.value = value


use_implicit_lreq = Bool()
use_implicit_lreq.set(True)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


def make_tuple(x, n):
    if is_sequence(x):
        return x
    return tuple([x for _ in range(n)])


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2.0), lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.gain = gain
        self.lrmul = lrmul
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.in_features) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input):
        if not self.implicit_lreq:
            bias = self.bias
            if bias is not None:
                bias = bias * self.lrmul
            return F.linear(input, self.weight * self.std, bias)
        else:
            return F.linear(input, self.weight, self.bias)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=use_implicit_lreq):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 2)
        self.stride = make_tuple(stride, 2)
        self.padding = make_tuple(padding, 2)
        self.output_padding = make_tuple(output_padding, 2)
        self.dilation = make_tuple(dilation, 2)
        self.groups = groups
        self.gain = gain
        self.lrmul = lrmul
        self.transpose = transpose
        self.fan_in = np.prod(self.kernel_size) * in_channels // groups
        self.transform_kernel = transform_kernel
        if transpose:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *self.kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.fan_in)
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x):
        if self.transpose:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv_transpose2d(x, w * self.std, bias, stride=self.stride,
                                          padding=self.padding, output_padding=self.output_padding,
                                          dilation=self.dilation, groups=self.groups)
            else:
                return F.conv_transpose2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                          output_padding=self.output_padding, dilation=self.dilation,
                                          groups=self.groups)
        else:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv2d(x, w * self.std, bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            else:
                return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)


class ConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transform_kernel=False, lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(ConvTranspose2d, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              output_padding=output_padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              gain=gain,
                                              transpose=True,
                                              transform_kernel=transform_kernel,
                                              lrmul=lrmul,
                                              implicit_lreq=implicit_lreq)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0), transpose=False):
        super(SeparableConv2d, self).__init__()
        self.spatial_conv = Conv2d(in_channels, in_channels, kernel_size, stride, padding, output_padding, dilation,
                                   in_channels, False, 1, transpose)
        self.channel_conv = Conv2d(in_channels, out_channels, 1, bias, 1, gain=gain)

    def forward(self, x):
        return self.channel_conv(self.spatial_conv(x))


class SeparableConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0)):
        super(SeparableConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, dilation, bias, gain, True)

