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
#import lreq as ln
from torch import nn as ln


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


class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False):
        super(EncodeBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1)
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=True)
        self.blur = Blur(inputs)
        self.last = last
        self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1)
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


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return x * (style[:, 0] + 1) + style[:, 1]


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.style_1 = ln.Linear(latent_size, 2 * (inputs if has_first_conv else outputs))
        if has_first_conv:
            self.conv_1 = ln.ConvTranspose2d(inputs, outputs, 3, 2, 1, output_padding=1)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.normal_(0.1, 0.02)
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=True)
        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.normal_(0.1, 0.02)
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=True)
        self.style_2 = ln.Linear(latent_size, 2 * outputs)
        self.blur = Blur(outputs)

    def forward(self, x, styles, noise):
        s = styles.pop()
        
        #x = style_mod(x, self.style_1(s))
        x = x * s[1] + s[0]
        
        if self.has_first_conv:
            x = self.blur(self.conv_1(x))
        
        if noise > 0.0:
            x = x + self.noise_weight_1 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]) * noise

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)

        s = styles.pop()
        #x = style_mod(x, self.style_2(s))
        x = x * s[1] + s[0]

        x = self.conv_2(x)
        
        if noise > 0.0:
            x = x + self.noise_weight_2 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]) * noise

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)
        
        return x

class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)
        self.instance_norm = nn.InstanceNorm2d(outputs, affine=True)
        
    def forward(self, x, styles):
        x = self.from_rgb(x)
        
        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        styles.append((m, std))
        
        x = self.instance_norm(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0)#, gain=1)
        #self.style_1 = ln.Linear(latent_size, 2 * inputs)
        
    def forward(self, x, styles):
        s = styles.pop()
        #x = style_mod(x, self.style_1(s))
        x = x * s[1] + s[0]
        x = self.to_rgb(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Autoencoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.from_rgb = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))
            block = EncodeBlock(inputs, outputs, i == self.layer_count - 1)

            print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            setattr(self, "encode_block%d" % (i + 1), block)
            inputs = outputs
            mul *= 2

        #self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        #self.fc2 = nn.Linear(inputs * 4 * 4, zsize)
        #self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul = 2**(self.layer_count-1)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        inputs = min(self.maxf, startf * mul)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            if i == 0:
                const_size = outputs

            block = DecodeBlock(inputs, outputs, latent_size, i != 0)
            self.to_rgb.append(ToRGB(outputs, channels))

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            print("decode_block%d %s styles in: %dl out resolution: %d" % ((i + 1), millify(count_parameters(block)), outputs, resolution))
            setattr(self, "decode_block%d" % (i + 1), block)
            inputs = outputs
            mul //= 2

        self.const = Parameter(torch.Tensor(1, const_size, 4, 4))
        init.ones_(self.const)

    def encode(self, x, lod):
        styles = []

        x = self.from_rgb[self.layer_count - lod - 1](x, styles)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x, styles)

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = []

        x = self.from_rgb[self.layer_count - lod - 1](x, styles)
        x = F.leaky_relu(x, 0.2)
        x = getattr(self, "encode_block%d" % (self.layer_count - lod - 1 + 1))(x, styles)

        x_prev = F.interpolate(x_orig, x.shape[-1])

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev, styles)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = x * blend + x_prev * (1.0 - blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = getattr(self, "encode_block%d" % (i + 1))(x, styles)

        return styles

    #
    # def reparameterize(self, mu, logvar):
    #     if self.training:
    #         std = torch.exp(0.5 * logvar)
    #         eps = torch.randn_like(std)
    #         return eps.mul(std).add_(mu)
    #     else:
    #         return mu

    def decode(self, styles, lod, noise):
        x = self.const
        #x = x.repeat(styles[0].shape[0], 1, 1, 1)
        styles = styles[:]

        for i in range(lod + 1):
            x = getattr(self, "decode_block%d" % (i + 1))(x, styles, noise)

        x = self.to_rgb[lod](x, styles)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const
        #x = x.repeat(styles[0].shape[0], 1, 1, 1)
        styles = styles[:]

        for i in range(lod):
            x = getattr(self, "decode_block%d" % (i + 1))(x, styles, noise)

        x_prev = self.to_rgb[lod - 1](x, styles)

        x = getattr(self, "decode_block%d" % (lod + 1))(x, styles, noise)
        x = self.to_rgb[lod](x, styles)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = x * blend + x_prev * (1.0 - blend)

        return x

    def forward(self, x, lod, blend):
        if blend == 1:
            styles = self.encode(x, lod)
            return self.decode(styles, lod, 1.0)
        else:
            styles = self.encode2(x, lod, blend)
            return self.decode2(styles, lod, blend, 1.0)
    #
    # def _forward(self, styles, lod, blend):
    #     if blend == 1:
    #         return self.decode(styles, lod, 1.)
    #     else:
    #         return self.decode2(styles, lod, blend, 1.)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


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

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


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
        #x = self.batch_norm_1(x)
        x = F.leaky_relu(x, 0.2)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
        #x = self.batch_norm_2(x)
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

        self.fc2 = ln.Linear(inputs, 1)#, gain=1)

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

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class DiscriminatorSmall(nn.Module):
    def __init__(self, d=64, channels=3):
        super(DiscriminatorSmall, self).__init__()
        self.conv1 = nn.Conv2d(channels, d, 3, 1, 1)
        self.conv2 = nn.Conv2d(d, 2 * d, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(2 * d)
        self.conv3 = nn.Conv2d(2 * d, 4 * d, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(4 * d)
        self.conv4 = nn.Conv2d(4 * d, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, _1, _2, _3):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x)).mean(dim=[1, 2, 3])
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
