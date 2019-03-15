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


def resize2d(img, size):
    if img.shape[2] == size and img.shape[3] == size:
        return img
    with torch.no_grad():
        return F.adaptive_avg_pool2d(img.detach(), size)


class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(EncodeBlock, self).__init__()
        self.conv_1 = nn.Conv2d(inputs, inputs, 3, 1, 1)
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=True)
        self.blur = Blur(inputs)
        self.conv_2 = nn.Conv2d(inputs, outputs, 3, 2, 1)
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=True)
        
    def forward(self, x, styles):
        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        styles.append((m, std))

        x = self.instance_norm_1(x)
        
        x = self.conv_2(self.blur(x))
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        styles.append((m, std))

        x = self.instance_norm_2(x)
        
        return x


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(DecodeBlock, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(inputs, outputs, 3, 2, 1, output_padding=1)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=True)
        self.conv_2 = nn.Conv2d(outputs, outputs, 3, 1, 1)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=True)
        self.blur = Blur(outputs)
        
    def forward(self, x, styles):
        s = styles.pop()
        x = x * s[1] + s[0]

        x = self.blur(self.conv_1(x))
        x = x + self.noise_weight_1 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]])

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        s = styles.pop()
        x = x * s[1] + s[0]

        x = self.conv_2(x)
        x = x + self.noise_weight_2 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]])

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_2(x)
        
        return x


class VAE(nn.Module):
    def __init__(self, zsize, maxf=256, layer_count=3, channels=3):
        super(VAE, self).__init__()
        self.maxf = maxf

        d = 64
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        self.from_rgb = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = d
        for i in range(self.layer_count):
            outputs = min(self.maxf, d * mul)

            self.from_rgb.append(nn.Conv2d(channels, inputs, 1, 1, 0))
            block = EncodeBlock(inputs, outputs)

            print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            setattr(self, "encode_block%d" % (i + 1), block)
            inputs = outputs
            mul *= 2

        self.d_max = inputs

        #self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        #self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

        #self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul //= 4

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 4

        for i in range(self.layer_count):
            outputs = min(self.maxf, d * mul)

            block = DecodeBlock(inputs, outputs)
            self.to_rgb.append(nn.Conv2d(outputs, channels, 1, 1, 0))

            print("decode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            setattr(self, "decode_block%d" % (i + 1), block)
            inputs = outputs
            mul //= 2

            resolution *= 2
            self.layer_to_resolution[i] = resolution

        self.const = Parameter(torch.Tensor(1, self.d_max, 4, 4))
        init.normal_(self.const)

    def encode(self, x, lod):
        styles = []

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x, styles)

        return styles

    def encode2(self, x, x_prev, lod, blend):
        styles = []

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = getattr(self, "encode_block%d" % (self.layer_count - lod - 1 + 1))(x, styles)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = x * blend + x_prev * (1.0 - blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x, styles)

        return styles

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, styles, lod):
        x = self.const

        styles = styles[:]

        for i in range(lod + 1):
            x = getattr(self, "decode_block%d" % (i + 1))(x, styles)

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend):
        x = self.const

        styles = styles[:]

        for i in range(lod):
            x = getattr(self, "decode_block%d" % (i + 1))(x, styles)

        x_prev = self.to_rgb[lod - 1](x)

        x = getattr(self, "decode_block%d" % (lod + 1))(x, styles)
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = x * blend + x_prev * (1.0 - blend)

        return x

    def forward(self, x, x_prev, lod, blend):
        if blend == 1:
            styles = self.encode(x, lod)
            return self.decode(styles, lod)
        else:
            styles = self.encode2(x, x_prev, lod, blend)
            x = self.decode2(styles, lod, blend)
            return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = nn.Conv2d(inputs, inputs, 3, 1, 1)
        self.blur = Blur(inputs)
        self.conv_2 = nn.Conv2d(inputs, outputs, 3, 2, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv_2(self.blur(x))
        x = F.leaky_relu(x, 0.2)

        return x


def minibatch_stddev_layer(x, group_size=4):
    y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt((y**2).mean(dim=0) + 1e-8).mean(dim=[1, 2, 3], keepdim=True)
    y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
    return torch.cat([x, y], dim=1)


class Discriminator(nn.Module):
    def __init__(self, zsize, maxf=256, layer_count=3, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf

        d = 64
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        self.from_rgb = nn.ModuleList()

        mul = 2
        inputs = d
        for i in range(self.layer_count):
            outputs = min(self.maxf, d * mul)

            self.from_rgb.append(nn.Conv2d(channels, inputs, 1, 1, 0))
            block = DiscriminatorBlock(inputs, outputs)

            print("encode_block%d %s" % ((i + 1), millify(count_parameters(block))))
            setattr(self, "encode_block%d" % (i + 1), block)
            inputs = outputs
            mul *= 2

        self.d_max = inputs

        self.conv = nn.Conv2d(inputs + 1, inputs, 3, 1, 1)
        self.fc1 = nn.Linear(inputs * 4 * 4, inputs)
        self.fc2 = nn.Linear(inputs, 1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x)

        x = minibatch_stddev_layer(x)

        x = F.leaky_relu(self.conv(x), 0.2)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)

        return x

    def encode2(self, x, x_prev, lod, blend):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = getattr(self, "encode_block%d" % (self.layer_count - lod - 1 + 1))(x)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = x * blend + x_prev * (1.0 - blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x)

        x = minibatch_stddev_layer(x)

        x = F.leaky_relu(self.conv(x), 0.2)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)

        return x

    def forward(self, x, x_prev, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, x_prev, lod, blend)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
