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

def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)

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
                 implicit_lreq=use_implicit_lreq,initial_weight=None):
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
        self.initial_weight = initial_weight
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
        self.std = self.gain / np.sqrt(self.fan_in) *self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            if self.initial_weight:
                self.weight = self.initial_weight
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

class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=use_implicit_lreq,initial_weight=None):
        super(Conv3d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 3) if isinstance(kernel_size,int) else kernel_size
        self.stride = make_tuple(stride, 3) if isinstance(stride,int) else stride
        self.padding = make_tuple(padding, 3) if isinstance(padding,int) else padding
        self.output_padding = make_tuple(output_padding, 3) if isinstance(output_padding,int) else output_padding
        self.dilation = make_tuple(dilation, 3) if isinstance(dilation,int) else dilation
        self.groups = groups
        self.gain = gain
        self.lrmul = lrmul
        self.transpose = transpose
        self.fan_in = np.prod(self.kernel_size) * in_channels // groups
        self.transform_kernel = transform_kernel
        self.initial_weight = initial_weight
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
        self.std = self.gain / np.sqrt(self.fan_in) *self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            if self.initial_weight:
                self.weight = self.initial_weight
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
                w = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant')
                w = w[:, :, 1:, 1:, 1:] + w[:, :, :-1, 1:, 1:] + w[:, :, 1:, :-1, 1:] + w[:, :, :-1, :-1, 1:] + w[:, :, 1:, 1:, :-1] + w[:, :, :-1, 1:, :-1] + w[:, :, 1:, :-1, :-1] + w[:, :, :-1, :-1, :-1]
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv_transpose3d(x, w * self.std, bias, stride=self.stride,
                                          padding=self.padding, output_padding=self.output_padding,
                                          dilation=self.dilation, groups=self.groups)
            else:
                return F.conv_transpose3d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                          output_padding=self.output_padding, dilation=self.dilation,
                                          groups=self.groups)
        else:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = (w[:, :, 1:, 1:, 1:] + w[:, :, :-1, 1:, 1:] + w[:, :, 1:, :-1, 1:] + w[:, :, :-1, :-1, 1:] + w[:, :, 1:, 1:, :-1] + w[:, :, :-1, 1:, :-1] + w[:, :, 1:, :-1, :-1] + w[:, :, :-1, :-1, :-1]) * 0.125
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv3d(x, w * self.std, bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            else:
                return F.conv3d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)

class StyleConv2dtest(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, latent_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=False,initial_weight=None,demod=True,upsample=False,temporal_w=False):
        super(StyleConv2dtest,self).__init__()
        self.demod = demod
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                                        groups=groups, bias=False, gain=gain, transpose=transpose, 
                                        transform_kernel=transform_kernel, lrmul=lrmul,
                                        implicit_lreq=implicit_lreq,initial_weight=initial_weight)
        self.style = Linear(latent_size, 2*in_channels, gain=1)
        if demod:
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, eps=1e-8)
        self.upsample = upsample
        self.transpose = transpose
        if bias:
            self.bias = Parameter(torch.Tensor(1,out_channels,1,1))
            with torch.no_grad():
                self.bias.zero_()
        if upsample:
            self.blur = Blur(out_channels)
        self.noise_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, style,noise=None):
        if self.upsample and not self.transpose:
            x = upscale2d(x)
        w = self.style(style)
        w = w.view(w.shape[0], 2, x.shape[1], 1, 1)
        x = w[:,1]+x*(w[:,0]+1)
        x = F.leaky_relu(self.conv(x),0.2)
        if self.demod:
            x = self.norm(x)
        x = self.bias+x
        if self.upsample:
            x = self.blur(x)
        if noise:
            x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight,
                                tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        
        return x
            
                 
class StyleConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, latent_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=False,initial_weight=None,demod=True,upsample=False,temporal_w=False):
        super(StyleConv2d,self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
                                        groups=groups, bias=bias, gain=gain, transpose=upsample, 
                                        transform_kernel=transform_kernel, lrmul=lrmul,
                                        implicit_lreq=implicit_lreq,initial_weight=initial_weight)
        self.demod=demod
        self.upsample = upsample
        self.transpose = upsample
        self.temporal_w = temporal_w
        if upsample:
            self.blur = Blur(out_channels)
        if temporal_w:
            self.modulation = Conv1d(latent_size, in_channels,1,1,0, gain=1)
        else:
            self.modulation = Linear(latent_size, in_channels, gain=1)
        self.noise_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, style,noise=None):
        batch, in_channels, height, width = x.shape
        if not self.temporal_w:
            assert style.dim()==2, "Style dimension not mach temporal_w condition"
        else:
            assert style.dim()==3, "Style dimension not mach temporal_w condition"
        style = self.modulation(style).view(batch, 1, in_channels, 1, 1)
        w = self.weight
        w = w if self.implicit_lreq else (w * self.std)
        if self.transpose:
            w = w.transpose(0,1) # out, in, H, W
        if not self.temporal_w:
            w2 = w[None, :, :, :, :] # batch, out_chan, in_chan, H, w 
            w = w2 * (1 + style)
            if self.demod:
                d = torch.rsqrt((w ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
                w = w * d
            _, _, _, *ws = w.shape
            if self.transpose:
                w = w.transpose(1,2).reshape(batch* in_channels, self.out_channels, *ws)
            else:
                w = w.view( batch * self.out_channels, in_channels,*ws)
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = w[..., 1:, 1:] + w[..., :-1, 1:] + w[..., 1:, :-1] + w[..., :-1, :-1]
                if not self.transpose:
                    w =w*0.25
            x = x.view(1, batch * in_channels, height, width)

            bias = self.bias
            if not self.implicit_lreq:
                if bias is not None:
                    bias = bias * self.lrmul
            if self.transpose:
                out = F.conv_transpose2d(x, w, None, stride=self.stride,
                                        padding=self.padding, output_padding=self.output_padding,
                                        dilation=self.dilation, groups=batch)
            else:
                out = F.conv2d(x, w, None, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=batch)
                
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channels, height, width)
            if bias is not None:
                out = out + bias[None,:,None,None]
            if self.upsample:
                out = self.blur(out)
            
            

        else:
            assert style.dim()==3, "Style dimension not mach temporal_w condition"
            raise ValueError('temporal_w is not support yet')

        if noise:
            out = torch.addcmul(out, value=1.0, tensor1=self.noise_weight,
                                  tensor2=torch.randn([out.shape[0], 1, out.shape[2], out.shape[3]]))
        return out

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=use_implicit_lreq, bias_initial = 0.):
        super(Conv1d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 1)
        self.stride = make_tuple(stride, 1)
        self.padding = make_tuple(padding, 1)
        self.output_padding = make_tuple(output_padding, 1)
        self.bias_initial = bias_initial
        self.dilation = make_tuple(dilation, 1)
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
        self.std = self.gain / np.sqrt(self.fan_in) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                nn.init.constant_(self.bias,self.bias_initial)

    def forward(self, x):
        if self.transpose:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1), mode='constant')
                w = w[:, :, 1:] + w[:, :, :-1]
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv_transpose1d(x, w * self.std, bias, stride=self.stride,
                                          padding=self.padding, output_padding=self.output_padding,
                                          dilation=self.dilation, groups=self.groups)
            else:
                return F.conv_transpose1d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                          output_padding=self.output_padding, dilation=self.dilation,
                                          groups=self.groups)
        else:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1), mode='constant')
                w = (w[:, :, 1:] + w[:, :, :-1]) * 0.5
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv1d(x, w * self.std, bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            else:
                return F.conv1d(x, w, self.bias, stride=self.stride, padding=self.padding,
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

class ConvTranspose1d(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transform_kernel=False, lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(ConvTranspose1d, self).__init__(in_channels=in_channels,
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

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0), transpose=False):
        super(SeparableConv1d, self).__init__()
        self.spatial_conv = Conv1d(in_channels, in_channels, kernel_size, stride, padding, output_padding, dilation,
                                   in_channels, False, 1, transpose)
        self.channel_conv = Conv1d(in_channels, out_channels, 1, bias, 1, gain=gain)

    def forward(self, x):
        return self.channel_conv(self.spatial_conv(x))

class SeparableConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0)):
        super(SeparableConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, dilation, bias, gain, True)

class SeparableConvTranspose1d(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0)):
        super(SeparableConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, dilation, bias, gain, True)


