# Copyright 2019-2020 Stanislav Pidhorskyi
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

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import lreq as ln
import math
from registry import *


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


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


class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False, fused_scale=True):
        super(EncodeBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)
        # self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = Blur(inputs)
        self.last = last
        self.fused_scale = fused_scale
        if last:
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)
        self.style_1 = ln.Linear(2 * inputs, latent_size)
        if last:
            self.style_2 = ln.Linear(outputs, latent_size)
        else:
            self.style_2 = ln.Linear(2 * outputs, latent_size)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)

        x = self.instance_norm_1(x)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))

            x = F.leaky_relu(x, 0.2)
            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(x.view(x.shape[0], x.shape[1]))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2

            x = F.leaky_relu(x, 0.2)

            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
            style_2 = torch.cat((m, std), dim=1)

            x = self.instance_norm_2(x)

            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))

        return x, w1, w2


class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=True, dense=False):
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        self.dense_ = dense
        self.fused_scale = fused_scale
        if self.dense_:
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.dense_:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)

        return x


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True, layer=0):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.fused_scale = fused_scale
        if has_first_conv:
            if fused_scale:
                self.conv_1 = ln.ConvTranspose2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_1 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_1 = ln.Linear(latent_size, 2 * outputs, gain=1)

        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_2 = ln.Linear(latent_size, 2 * outputs, gain=1)

        self.layer = layer

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, s1, s2, noise):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)

        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8
        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        x = style_mod(x, self.style_1(s1))

        x = self.conv_2(x)

        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x +  s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        return x


class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0, gain=0.03)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


@ENCODERS.register("EncoderDefault")
class Encoder_old(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Encoder_old, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, False, fused_scale=fused_scale)

            resolution //= 2

            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


@ENCODERS.register("EncoderWithFC")
class EncoderWithFC(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(EncoderWithFC, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = ln.Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles, self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles, self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


@ENCODERS.register("EncoderWithStatistics")
class Encoder(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Encoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


@ENCODERS.register("EncoderNoStyle")
class EncoderNoStyle(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=512, channels=3):
        super(EncoderNoStyle, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = DiscriminatorBlock(inputs, outputs, last=False, fused_scale=fused_scale, dense=i == self.layer_count - 1)

            resolution //= 2

            #print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = ln.Linear(inputs, latent_size, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x).view(x.shape[0], 1, x.shape[1])

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x).view(x.shape[0], 1, x.shape[1])

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


@DISCRIMINATORS.register("DiscriminatorDefault")
class Discriminator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = DiscriminatorBlock(inputs, outputs, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            #print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = ln.Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


@GENERATORS.register("GeneratorDefault")
class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size

        mul = 2 ** (self.layer_count - 1)

        inputs = min(self.maxf, startf * mul)
        self.const = Parameter(torch.Tensor(1, inputs, 4, 4))
        init.ones_(self.const)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()

        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale, layer=i)

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels))

            #print("decode_block%d %s styles in: %dl out resolution: %d" % (
            #    (i + 1), millify(count_parameters(block)), outputs, resolution))
            self.decode_block.append(block)
            inputs = outputs
            mul //= 2

        self.to_rgb = to_rgb

    def decode(self, styles, lod, noise):
        x = self.const

        for i in range(lod + 1):
            x = self.decode_block[i](x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const

        for i in range(lod):
            x = self.decode_block[i](x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)

        x_prev = self.to_rgb[lod - 1](x)

        x = self.decode_block[lod](x, styles[:, 2 * lod + 0], styles[:, 2 * lod + 1], noise)
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = torch.lerp(x_prev, x, blend)

        return x

    def forward(self, styles, lod, blend, noise):
        if blend == 1:
            return self.decode(styles, lod, noise)
        else:
            return self.decode2(styles, lod, blend, noise)

    def get_statistics(self, lod):
        rgb_std = self.to_rgb[lod].to_rgb.weight.std().item()
        rgb_std_c = self.to_rgb[lod].to_rgb.std

        layers = []
        for i in range(lod + 1):
            conv_1 = 1.0
            conv_1_c = 1.0
            if i != 0:
                conv_1 = self.decode_block[i].conv_1.weight.std().item()
                conv_1_c = self.decode_block[i].conv_1.std
            conv_2 = self.decode_block[i].conv_2.weight.std().item()
            conv_2_c = self.decode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 24

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


@GENERATORS.register("DCGANGenerator")
class DCGANGenerator(nn.Module):
    def __init__(self):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128, nc, 4, 2, 1),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x.view(x.shape[0], nz, 1, 1))


@ENCODERS.register("DCGANEncoder")
class DCGANEncoder(nn.Module):
    def __init__(self):
        super(DCGANEncoder, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(256, 24, 4, 1, 0),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(x.shape[0], x.shape[1])


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = ln.Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


@MAPPINGS.register("MappingDefault")
class Mapping(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(Mapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.01)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))

    def forward(self, z):
        x = pixel_norm(z)

        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)


@MAPPINGS.register("MappingToLatent")
class VAEMappingToLatent_old(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(VAEMappingToLatent_old, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = 2 * dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = ln.Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x.view(x.shape[0], 2, x.shape[2] // 2)


@MAPPINGS.register("MappingToLatentNoStyle")
class VAEMappingToLatentNoStyle(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(VAEMappingToLatentNoStyle, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = ln.Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            if i == self.mapping_layers - 1:
                #x = self.map_blocks[i](x)
                x = self.map_blocks[i](x)
            else:
                #x = self.map_blocks[i](x)
                x = self.map_blocks[i](x)
        return x


@MAPPINGS.register("MappingFromLatent")
class VAEMappingFromLatent(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(VAEMappingFromLatent, self).__init__()
        inputs = dlatent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = latent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))

    def forward(self, x):
        x = pixel_norm(x)

        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)


@ENCODERS.register("EncoderFC")
class EncoderFC(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(EncoderFC, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        self.fc_1 = ln.Linear(28 * 28, 1024)
        self.fc_2 = ln.Linear(1024, 1024)
        self.fc_3 = ln.Linear(1024, latent_size)

    def encode(self, x, lod):
        x = F.interpolate(x, 28)
        x = x.view(x.shape[0], 28 * 28)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)
        x = F.leaky_relu(x, 0.2)

        return x

    def forward(self, x, lod, blend):
        return self.encode(x, lod)


@GENERATORS.register("GeneratorFC")
class GeneratorFC(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(GeneratorFC, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        self.fc_1 = ln.Linear(latent_size, 1024)
        self.fc_2 = ln.Linear(1024, 1024)
        self.fc_3 = ln.Linear(1024, 28 * 28)

        self.layer_to_resolution = [28] * 10

    def decode(self, x, lod, blend_factor, noise):
        if len(x.shape) == 3:
            x = x[:, 0]  # no styles
        x.view(x.shape[0], self.latent_size)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, 28, 28)
        x = F.interpolate(x, 2 ** (2 + lod))
        return x

    def forward(self, x, lod, blend_factor, noise):
        return self.decode(x, lod, blend_factor, noise)
