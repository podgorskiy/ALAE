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
    def __init__(self, inputs, outputs, latent_size, last=False):
        super(EncodeBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1)
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=True)
        self.blur = Blur(inputs)
        self.last = last
        self.conv_2 = ln.Conv2d(inputs, outputs, 3, 2, 1)
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=True)
        self.style_1 = ln.Linear(2 * inputs, latent_size)
        self.style_2 = ln.Linear(2 * outputs, latent_size)

    def encode_styles(self, style_1, style_2):
        w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
        w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))
        return w1 + w2

    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)

        x = self.instance_norm_1(x)
        
        x = self.conv_2(self.blur(x))
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_2 = torch.cat((m, std), dim=1)

        x = self.instance_norm_2(x)

        return x, style_1, style_2


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

    def decode_styles(self, w):
        style_vec1 = self.style_1(w)
        style_vec2 = self.style_2(w)
        return style_vec1, style_vec2

    def forward(self, x, style_1, style_2):
        x = style_mod(x, style_1)
        #x = x * s[1] + s[0]
        
        if self.has_first_conv:
            x = self.blur(self.conv_1(x))
        
        #if noise > 0.0:
        x = x + self.noise_weight_1 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]])

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)

        x = style_mod(x, style_2)
        #x = x * s[1] + s[0]

        x = self.conv_2(x)
        
        #if noise > 0.0:
        x = x + self.noise_weight_2 * torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]])

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)
        
        return x


class FromRGB(nn.Module):
    def __init__(self, channels, outputs, latent_size):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)
        self.instance_norm = nn.InstanceNorm2d(outputs, affine=True)
        self.style_1 = ln.Linear(2 * outputs, latent_size)

    def encode_styles(self, style):
        w = self.style_1(style.view(style.shape[0], style.shape[1]))
        return w

    def forward(self, x):
        x = self.from_rgb(x)
        
        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style = torch.cat((m, std), dim=1)
        
        x = self.instance_norm(x)
        return x, style


class ToRGB(nn.Module):
    def __init__(self, inputs, channels, latent_size):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0)#, gain=1)
        self.style_1 = ln.Linear(latent_size, 2 * inputs)

    def decode_styles(self, w):
        style_vec1 = self.style_1(w)
        return style_vec1

    def forward(self, x, style):
        x = style_mod(x, style)
        #x = x * s[1] + s[0]
        x = self.to_rgb(x)
        return x


class Encoder(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Encoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.from_rgb = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs, 2 * latent_size))
            block = EncodeBlock(inputs, outputs, 2 * latent_size, i == self.layer_count - 1)

            print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            setattr(self, "encode_block%d" % (i + 1), block)
            inputs = outputs
            mul *= 2

        mul = 2**(self.layer_count-1)

        self.style_sizes = []

        inputs = min(self.maxf, startf * mul)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            has_first_conv = i != 0
            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]
            inputs = outputs
            mul //= 2

        self.style_sizes += [2 * startf]
        self.dense_style_encode = ln.Linear(sum(self.style_sizes), latent_size)

    def style_encode(self, styles, lod):
        s = styles.pop(0)
        w = self.from_rgb[self.layer_count - lod - 1].encode_styles(s)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            s1 = styles.pop(0)
            s2 = styles.pop(0)
            w += getattr(self, "encode_block%d" % (i + 1)).encode_styles(s1, s2)
        return w

    def style_encode2(self, styles, lod, blend):
        s = styles.pop(0)
        w = self.from_rgb[self.layer_count - lod - 1].encode_styles(s) * blend

        s1 = styles.pop(0)
        s2 = styles.pop(0)
        w += getattr(self, "encode_block%d" % (self.layer_count - lod - 1 + 1)).encode_styles(s1, s2) * blend

        s = styles.pop(0)
        w += self.from_rgb[self.layer_count - (lod - 1) - 1].encode_styles(s)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            s1 = styles.pop(0)
            s2 = styles.pop(0)
            w += getattr(self, "encode_block%d" % (i + 1)).encode_styles(s1, s2)
        return w

    def encode(self, x, lod):
        styles = []

        x, style = self.from_rgb[self.layer_count - lod - 1](x)
        styles += [style]
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = getattr(self, "encode_block%d" % (i + 1))(x)
            styles += [s1, s2]

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = []

        x, style = self.from_rgb[self.layer_count - lod - 1](x)
        styles += [style]
        x = F.leaky_relu(x, 0.2)
        x, s1, s2 = getattr(self, "encode_block%d" % (self.layer_count - lod - 1 + 1))(x)
        styles += [s1, s2]

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev, style = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        styles += [style]
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = x * blend + x_prev * (1.0 - blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = getattr(self, "encode_block%d" % (i + 1))(x)
            styles += [s1, s2]

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            styles = self.encode(x, lod)
            w = self.style_encode(styles, lod)

            mu, logvar = torch.split(w, [self.latent_size, self.latent_size], dim=1)

            return mu, logvar
        else:
            styles = self.encode2(x, lod, blend)
            w = self.style_encode2(styles, lod, blend)

            mu, logvar = torch.split(w, [self.latent_size, self.latent_size], dim=1)

            return mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Decoder(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Decoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.to_rgb = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2**(self.layer_count-1)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        inputs = min(self.maxf, startf * mul)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            if i == 0:
                const_size = outputs

            has_first_conv = i != 0
            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv)

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            self.to_rgb.append(ToRGB(outputs, channels, latent_size))

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            print("decode_block%d %s styles in: %dl out resolution: %d" % ((i + 1), millify(count_parameters(block)), outputs, resolution))
            setattr(self, "decode_block%d" % (i + 1), block)
            inputs = outputs
            mul //= 2

        self.style_sizes += [2 * startf]

        self.dense_style_decode = ln.Linear(latent_size, sum(self.style_sizes))

        self.const = Parameter(torch.Tensor(1, const_size, 4, 4))
        init.ones_(self.const)

    def style_decode(self, w, lod):
        styles = []

        for i in range(lod + 1):
            s1, s2 = getattr(self, "decode_block%d" % (i + 1)).decode_styles(w)
            styles += [s1, s2]

        s = self.to_rgb[lod].decode_styles(w)
        styles += [s]
        return styles

    def style_decode2(self, w, lod, blend):
        styles = []

        for i in range(lod):
            s1, s2 = getattr(self, "decode_block%d" % (i + 1)).decode_styles(w)
            styles += [s1, s2]

        s = self.to_rgb[lod - 1].decode_styles(w)
        styles += [s]

        s1, s2 = getattr(self, "decode_block%d" % (lod + 1)).decode_styles(w)
        styles += [s1, s2]

        s = self.to_rgb[lod].decode_styles(w)
        styles += [s]

        return styles

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, styles, lod, noise):
        x = self.const
        styles = styles[:]

        for i in range(lod + 1):
            s1 = styles.pop(0)
            s2 = styles.pop(0)
            x = getattr(self, "decode_block%d" % (i + 1))(x, s1, s2)

        s = styles.pop(0)
        x = self.to_rgb[lod](x, s)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const
        #x = x.repeat(styles[0].shape[0], 1, 1, 1)
        styles = styles[:]

        for i in range(lod):
            s1 = styles.pop(0)
            s2 = styles.pop(0)
            x = getattr(self, "decode_block%d" % (i + 1))(x, s1, s2)

        s = styles.pop(0)
        x_prev = self.to_rgb[lod - 1](x, s)

        s1 = styles.pop(0)
        s2 = styles.pop(0)
        x = getattr(self, "decode_block%d" % (lod + 1))(x, s1, s2)
        s = styles.pop(0)
        x = self.to_rgb[lod](x, s)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = x * blend + x_prev * (1.0 - blend)

        return x

    def forward(self, mu, logvar, lod, blend):
        if blend == 1:
            w = self.reparameterize(mu, logvar)

            styles = self.style_decode(w, lod)
            return self.decode(styles, lod, 1.0)
        else:
            w = self.reparameterize(mu, logvar)

            styles = self.style_decode2(w, lod, blend)
            return self.decode2(styles, lod, blend, 1.0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
