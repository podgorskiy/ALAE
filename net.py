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
from dlutils.pytorch import count_parameters, millify
#from torch import nn as ln
import lreq as ln


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


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


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return x * (style[:, 0] + 1) + style[:, 1]


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        if has_first_conv:
            self.conv_1 = ln.ConvTranspose2d(inputs, outputs, 3, 2, 1, output_padding=1)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.normal_(0.1, 0.02)
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=True)
        self.style_1 = ln.Linear(latent_size, 2 * outputs)#, gain=1)
        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.normal_(0.1, 0.02)
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=True)
        self.style_2 = ln.Linear(latent_size, 2 * outputs)
        self.blur = Blur(outputs)

    def forward(self, x, styles, noise):
        if self.has_first_conv:
            x = self.blur(self.conv_1(x))
        
        if noise > 0.0:
            x = x + self.noise_weight_1 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]) * noise

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)

        s = styles.pop()
        
        x = style_mod(x, self.style_1(s))

        x = self.conv_2(x)
        
        if noise > 0.0:
            x = x + self.noise_weight_2 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]) * noise

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)
        
        s = styles.pop()
        x = style_mod(x, self.style_2(s))

        return x


class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.to_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2**(self.layer_count-1)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        inputs = min(self.maxf, startf * mul)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            if i == 0:
                const_size = outputs

            block = DecodeBlock(inputs, outputs, latent_size, i != 0)
            self.to_rgb.append(ln.Conv2d(outputs, channels, 1, 1, 0, gain=1))

            print("decode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            setattr(self, "decode_block%d" % (i + 1), block)
            inputs = outputs
            mul //= 2

            resolution *= 2
            self.layer_to_resolution[i] = resolution

        self.const = Parameter(torch.Tensor(1, const_size, 4, 4))
        init.ones_(self.const)

    def decode(self, styles, lod, noise):
        x = self.const
        x = x.repeat(styles[0].shape[0], 1, 1, 1)
        styles = styles[:]

        for i in range(lod + 1):
            x = getattr(self, "decode_block%d" % (i + 1))(x, styles, noise)

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const
        x = x.repeat(styles[0].shape[0], 1, 1, 1)
        styles = styles[:]

        for i in range(lod):
            x = getattr(self, "decode_block%d" % (i + 1))(x, styles, noise)

        x_prev = self.to_rgb[lod - 1](x)

        x = getattr(self, "decode_block%d" % (lod + 1))(x, styles, noise)
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = x * blend + x_prev * (1.0 - blend)

        return x

    def forward(self, styles, lod, blend):
        if blend == 1:
            return self.decode(styles, lod, 1.)
        else:
            return self.decode2(styles, lod, blend, 1.)


def minibatch_stddev_layer(x, group_size=4):
    group_size = min(group_size, x.shape[0])
    size = x.shape[0]
    if x.shape[0] % group_size != 0:
        x = torch.cat([x, x[:(group_size - (x.shape[0] % group_size)) % group_size]])
    y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt((y**2).mean(dim=0) + 1e-8).mean(dim=[1, 2, 3], keepdim=True)
    y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
    return torch.cat([x, y], dim=1)[:size]


class MappingBlock(nn.Module):
    def __init__(self, inputs, output):
        super(MappingBlock, self).__init__()
        self.fc = nn.Linear(inputs, output)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


class Mapping(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(Mapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)
            print("dense %d %s" % ((i + 1), millify(count_parameters(block))))

    def forward(self, z):
        x = pixel_norm(z)

        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)

        return list(x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1).split(1, 1))


class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False):
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1)
        self.blur = Blur(inputs)
        self.last = last
        if last:
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1)

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.2)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
        x = F.leaky_relu(x, 0.2)

        return x


class Discriminator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()

        mul = 2
        inputs = startf
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(ln.Conv2d(channels, inputs, 1, 1, 0))
            block = DiscriminatorBlock(inputs, outputs, i == self.layer_count - 1)

            print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            setattr(self, "encode_block%d" % (i + 1), block)
            inputs = outputs
            mul *= 2

        self.fc2 = ln.Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x)

        return self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = getattr(self, "encode_block%d" % (self.layer_count - lod - 1 + 1))(x)

        x_prev = F.interpolate(x_orig, x.shape[-1])

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = x * blend + x_prev * (1.0 - blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x)

        return self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)
