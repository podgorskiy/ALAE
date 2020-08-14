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
from torch.nn import Parameter as P
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import lreq as ln
import math
from registry import *

PARTIAL_SN = False
USE_SN = False
def sn(module,use_sn=USE_SN):
    if use_sn:
        return torch.nn.utils.spectral_norm(module)
    else:
        return module


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style, bias = True):
    if style.dim()==2:
        style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    elif style.dim()==3:
        style = style.view(style.shape[0], 2, x.shape[1], style.shape[2], 1)
    if bias:
        return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)
    else:
        return x*(style[:,0]+1)


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)

class Downsample(nn.Module):
    def __init__(self,scale_factor):
        super(Downsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self,x):
        return F.interpolate(x,scale_factor=1/self.scale_factor)

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

class AdaIN(nn.Module):
    def __init__(self, latent_size,outputs,temporal_w=False):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(outputs,affine=False, eps=1e-8)
        self.style = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0,gain=1)) if temporal_w else sn(ln.Linear(latent_size, 2 * outputs, gain=1))
    def forward(self,x,w):
        x = self.instance_norm(x)
        x = style_mod(x,self.style(w))
        return x

class INencoder(nn.Module):
    def __init__(self, inputs,latent_size,temporal_w=False):
        super(INencoder, self).__init__()
        self.temporal_w = temporal_w
        self.instance_norm = nn.InstanceNorm2d(inputs,affine=False)
        self.style = sn(ln.Conv1d(2 * inputs, latent_size,1,1,0)) if temporal_w else sn(ln.Linear(2 * inputs, latent_size))
    def forward(self,x):
        m = torch.mean(x, dim=[3] if self.temporal_w else [2,3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[3] if self.temporal_w else [2,3], keepdim=True))
        style = torch.cat((m,std),dim=1)
        x = self.instance_norm(x)
        if self.temporal_w:
            w = self.style(style.view(style.shape[0], style.shape[1],style.shape[2]))
        else:
            w = self.style(style.view(style.shape[0], style.shape[1]))
        return x,w

class Attention(nn.Module):
  def __init__(self, inputs,temporal_w=False,attentional_style=False,decoding=True,latent_size=None,heads=1):
    super(Attention, self).__init__()
    # Channel multiplier
    self.inputs = inputs
    self.temporal_w = temporal_w
    self.decoding = decoding
    self.attentional_style = attentional_style
    self.att_denorm = 8
    self.heads = heads
    self.theta = sn(ln.Conv2d(inputs, inputs // self.att_denorm, 1,1,0, bias=False))
    self.phi = sn(ln.Conv2d(inputs, inputs // self.att_denorm, 1,1,0, bias=False))
    self.g = sn(ln.Conv2d(inputs, inputs // 2, 1,1,0, bias=False))
    self.o = sn(ln.Conv2d(inputs // 2, inputs, 1,1,0, bias=False))
    if not attentional_style:
        self.norm_theta =  nn.InstanceNorm2d(inputs // self.att_denorm,affine=True)
        self.norm_phi = nn.InstanceNorm2d(inputs // self.att_denorm,affine=True)
        self.norm_g = nn.InstanceNorm2d(inputs // 2,affine=True)
    else:
        if decoding:
            self.norm_theta = AdaIN(latent_size,inputs//self.att_denorm,temporal_w=temporal_w)
            self.norm_phi = AdaIN(latent_size,inputs//self.att_denorm,temporal_w=temporal_w)
            self.norm_g = AdaIN(latent_size,inputs//2,temporal_w=temporal_w)
        else:
            self.norm_theta = INencoder(inputs//self.att_denorm,latent_size,temporal_w=temporal_w)
            self.norm_phi = INencoder(inputs//self.att_denorm,latent_size,temporal_w=temporal_w)
            self.norm_g = INencoder(inputs//2,latent_size,temporal_w=temporal_w)

    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)

  def forward(self, x, w=None):
    # Apply convs
    x = x.contiguous()
    theta = self.theta(x)
    theta = self.norm_theta(theta,w) if (self.attentional_style and self.decoding) else self.norm_theta(theta)
    phi = F.max_pool2d(self.phi(x), [2,2])
    phi = self.norm_phi(phi,w) if (self.attentional_style and self.decoding) else self.norm_phi(phi)
    g = F.max_pool2d(self.g(x), [2,2])    
    g = self.norm_g(g,w) if (self.attentional_style and self.decoding) else self.norm_g(g)
    if self.attentional_style and not self.decoding:
        theta,w_theta = theta
        phi,w_phi = phi
        g,w_g = g
        w = w_theta+w_phi+w_g

    # Perform reshapes
    self.theta_ = theta.reshape(-1, self.inputs // self.att_denorm//self.heads, self.heads ,x.shape[2] * x.shape[3])
    self.phi_ = phi.reshape(-1, self.inputs // self.att_denorm//self.heads, self.heads, x.shape[2] * x.shape[3] // 4)
    g = g.reshape(-1, self.inputs // 2//self.heads, self.heads, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    self.beta = F.softmax(torch.einsum('bchi,bchj->bhij',self.theta_, self.phi_), -1)
    # self.beta = F.softmax(torch.bmm(self.theta_, self.phi_), -1)
    # Attention map times g path
    o = self.o(torch.einsum('bchj,bhij->bchi',g, self.beta).reshape(-1, self.inputs // 2, x.shape[2], x.shape[3]))
    # o = self.o(torch.bmm(g, self.beta.transpose(1,2)).view(-1, self.inputs // 2, x.shape[2], x.shape[3]))
    return (self.gamma * o + x, w) if (self.attentional_style and (not self.decoding)) else self.gamma * o + x

class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False,islast=False, fused_scale=True,temporal_w=False,residual=False,resample=False,temporal_samples=None,spec_chans=None):
        super(EncodeBlock, self).__init__()
        self.conv_1 = sn(ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False))
        # self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = Blur(inputs)
        self.last = last
        self.islast = islast
        self.fused_scale = False if temporal_w else fused_scale
        self.residual = residual
        self.resample=resample
        self.temporal_w = temporal_w
        if last:
            if self.temporal_w:
                self.conv_2 = sn(ln.Conv2d(inputs * spec_chans, outputs, 3, 1, 1, bias=False))
            else:
                self.dense = sn(ln.Linear(inputs * temporal_samples * spec_chans, outputs))
        else:
            if resample and fused_scale:
                self.conv_2 = sn(ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True))
            else:
                self.conv_2 = sn(ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False))

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)

        if self.temporal_w:
            self.style_1 = sn(ln.Conv1d(2 * inputs, latent_size,1,1,0),use_sn=PARTIAL_SN)
            if last:
                self.style_2 = sn(ln.Conv1d(outputs, latent_size,1,1,0),use_sn=PARTIAL_SN)
            else:
                self.style_2 = sn(ln.Conv1d(2 * outputs, latent_size,1,1,0),use_sn=PARTIAL_SN)
        else:
            self.style_1 = sn(ln.Linear(2 * inputs, latent_size),use_sn=PARTIAL_SN)
            if last:
                self.style_2 = sn(ln.Linear(outputs, latent_size),use_sn=PARTIAL_SN)
            else:
                self.style_2 = sn(ln.Linear(2 * outputs, latent_size),use_sn=PARTIAL_SN)

        if residual and not islast:
            if inputs==outputs:
                if not resample:
                    self.skip = nn.Identity()
                else:
                    self.skip = Downsample(scale_factor=2)
            else:
                if not resample:
                    self.skip = nn.Sequential(
                                            sn(ln.Conv2d(inputs, outputs, 1, 1, 0, bias=False),use_sn=PARTIAL_SN),
                                            nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)
                    )
                else:
                    self.skip = nn.Sequential(
                                            sn(ln.Conv2d(inputs, outputs, 1, 2, 0, bias=False, transform_kernel=True),use_sn=PARTIAL_SN),
                                            nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)
                    )

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.residual:
            x = F.leaky_relu(x)
            x_input = x
        x = self.conv_1(x) + self.bias_1
        

        if self.temporal_w:
            m = torch.mean(x, dim=[3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m) ** 2, dim=[3], keepdim=True))
        else:
            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)

        x = self.instance_norm_1(x)
        x = F.leaky_relu(x, 0.2)

        if self.last:
            if self.temporal_w:
                x = self.conv_2(x.view(x.shape[0], -1,x.shape[2]))
            else:
                x = self.dense(x.view(x.shape[0], -1))

            x = F.leaky_relu(x, 0.2)
            if self.temporal_w:
                w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1],style_1.shape[2]))
                w2 = self.style_2(x.view(x.shape[0], x.shape[1],x.shape[2]))
            else:
                w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
                w2 = self.style_2(x.view(x.shape[0], x.shape[1]))
        else:
            x = self.conv_2(self.blur(x))
            x = x + self.bias_2

            
            if self.temporal_w:
                m = torch.mean(x, dim=[3], keepdim=True)
                std = torch.sqrt(torch.mean((x - m) ** 2, dim=[3], keepdim=True))
            else:
                m = torch.mean(x, dim=[2, 3], keepdim=True)
                std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
            style_2 = torch.cat((m, std), dim=1)

            x = self.instance_norm_2(x)
            if self.temporal_w:
                w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1],style_1.shape[2]))
                w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1],style_2.shape[2]))
            else:
                w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
                w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))
            if not self.fused_scale:
                x = downscale2d(x)

        if not self.islast:
            if self.residual:
                x = self.skip(x_input)+x
        else:
            x = F.leaky_relu(x, 0.2)    
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
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True, layer=0,temporal_w=False,residual=False,resample = False):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.temporal_w = temporal_w
        self.fused_scale = fused_scale
        self.residual =residual
        self.resample = resample
        if has_first_conv:
            if resample and fused_scale:
                self.conv_1 = sn(ln.ConvTranspose2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True),use_sn=PARTIAL_SN)
            else:
                self.conv_1 = sn(ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False),use_sn=PARTIAL_SN)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        if temporal_w:
            self.style_1 = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0, gain=1),use_sn=PARTIAL_SN)
            self.style_2 = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0, gain=1),use_sn=PARTIAL_SN)
        else: 
            self.style_1 = sn(ln.Linear(latent_size, 2 * outputs, gain=1),use_sn=PARTIAL_SN)
            self.style_2 = sn(ln.Linear(latent_size, 2 * outputs, gain=1),use_sn=PARTIAL_SN)

        self.conv_2 = sn(ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False),use_sn=PARTIAL_SN)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        
        if residual and has_first_conv:
            if inputs==outputs:
                if not resample:
                    self.skip = nn.Identity()
                else:
                    self.skip = nn.Upsample(scale_factor=2)
            else:
                if not resample:
                    self.skip = nn.Sequential(
                                            sn(ln.Conv2d(inputs, outputs, 1, 1, 0, bias=False),use_sn=PARTIAL_SN),
                                            nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)
                    )
                else:
                    self.skip = nn.Sequential(
                                            sn(ln.ConvTranspose2d(inputs, outputs, 1, 2, 0, bias=False, transform_kernel=True),use_sn=PARTIAL_SN),
                                            nn.InstanceNorm2d(outputs, affine=True, eps=1e-8)
                    )

        self.layer = layer

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, s1, s2, noise):
        if self.has_first_conv:
            if self.residual:
                x = F.leaky_relu(x)
                x_input = x
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

        x = self.instance_norm_1(x)
        x = style_mod(x, self.style_1(s1))

        x = F.leaky_relu(x, 0.2)

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

        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        if self.residual:
            if self.has_first_conv:
                x = self.skip(x_input)+x
        else:
            x = F.leaky_relu(x, 0.2)

        return x


class FromRGB(nn.Module):
    def __init__(self, channels, outputs,residual=False):
        super(FromRGB, self).__init__()
        self.residual=residual
        self.from_rgb = sn(ln.Conv2d(channels, outputs, 1, 1, 0))

    def forward(self, x):
        x = self.from_rgb(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    def __init__(self, inputs, channels,residual=False):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.residual = residual
        self.to_rgb = sn(ln.Conv2d(inputs, channels, 1, 1, 0, gain=0.03),use_sn=PARTIAL_SN)

    def forward(self, x):
        if self.residual:
            x = F.leaky_relu(x, 0.2)
        x = self.to_rgb(x)
        return x


@ENCODERS.register("EncoderDefault")
class Encoder_old(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3,average_w = False,temporal_w=False,residual=False,attention=None,temporal_samples=None,spec_chans=None,attentional_style=False,heads=1):
        super(Encoder_old, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size
        self.average_w = average_w
        self.temporal_w = temporal_w
        self.attentional_style = attentional_style
        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()
        self.attention_block = nn.ModuleList()
        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs,residual=residual))
            apply_attention = attention and attention[self.layer_count-i-1]
            non_local = Attention(inputs,temporal_w=temporal_w,attentional_style=attentional_style,decoding=False,latent_size=latent_size,heads=heads) if apply_attention else None
            self.attention_block.append(non_local)
            fused_scale = resolution >= 128
            current_spec_chans = spec_chans // 2**i
            current_temporal_samples = temporal_samples // 2**i
            islast = i==(self.layer_count-1)
            block = EncodeBlock(inputs, outputs, latent_size, False, islast, fused_scale=fused_scale,temporal_w=temporal_w,residual=residual,resample=True,temporal_samples=current_temporal_samples,spec_chans=current_spec_chans)

            resolution //= 2

            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        if self.temporal_w:
            styles = torch.zeros(x.shape[0], 1, self.latent_size,128)
        else:
            styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            if self.attention_block[i]:
                x = self.attention_block[i](x)
                if self.attentional_style:
                    x,s = x
            x, s1, s2 = self.encode_block[i](x)
            if self.temporal_w and i!=0:
                s1 = F.interpolate(s1,scale_factor=2**i)
                s2 = F.interpolate(s2,scale_factor=2**i)
            styles[:, 0] += s1 + s2 + (s if (self.attention_block[i] and self.attentional_style) else 0)
        if self.average_w:
            styles /= (lod+1)
        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        if self.temporal_w:
            styles = torch.zeros(x.shape[0], 1, self.latent_size,128)
        else:
            styles = torch.zeros(x.shape[0], 1, self.latent_size)
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        if self.attention_block[self.layer_count - lod - 1]:
            x = self.attention_block[self.layer_count - lod - 1](x)
            if self.attentional_style:
                x,s = x
        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        if self.temporal_w and i!=0:
            s1 = F.interpolate(s1,scale_factor=2**(layer_count - lod - 1))
            s2 = F.interpolate(s2,scale_factor=2**(layer_count - lod - 1))
        styles[:, 0] += s1 * blend + s2 * blend + (s*blend if (self.attention_block[self.layer_count - lod - 1] and self.attentional_style) else 0)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            if self.attention_block[i]:
                x = self.attention_block[i](x)
                if self.attentional_style:
                    x,s = x
            x, s1, s2 = self.encode_block[i](x)
            if self.temporal_w and i!=0:
                s1 = F.interpolate(s1,scale_factor=2**i)
                s2 = F.interpolate(s2,scale_factor=2**i)
            styles[:, 0] += s1 + s2 + (s if (self.attention_block[i] and self.attentional_style) else 0)
        if self.average_w:
            styles /= (lod+1)
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
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3, temporal_samples=128,spec_chans=128,temporal_w=False,init_zeros=False,residual=False,attention=None,attentional_style=False,heads=1):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size
        self.temporal_w = temporal_w
        self.init_zeros = init_zeros
        self.attention = attention
        self.attentional_style = attentional_style
        mul = 2 ** (self.layer_count - 1)

        inputs = min(self.maxf, startf * mul)
        init_specchans = spec_chans//2**(self.layer_count-1)
        init_temporalsamples = temporal_samples//2**(self.layer_count-1)
        self.const = Parameter(torch.Tensor(1, inputs, init_temporalsamples, init_specchans))
        if init_zeros:
            init.zeros_(self.const)
        else:
            init.ones_(self.const)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()
        self.attention_block = nn.ModuleList()
        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale, layer=i,temporal_w=temporal_w,residual=residual,resample=True)

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels,residual=residual))

            #print("decode_block%d %s styles in: %dl out resolution: %d" % (
            #    (i + 1), millify(count_parameters(block)), outputs, resolution))
            apply_attention = attention and attention[i]
            non_local = Attention(outputs,temporal_w=temporal_w,attentional_style=attentional_style,decoding=True,latent_size=latent_size,heads=heads) if apply_attention else None
            self.decode_block.append(block)
            self.attention_block.append(non_local)
            inputs = outputs
            mul //= 2

        self.to_rgb = to_rgb

    def decode(self, styles, lod, noise):
        x = self.const

        for i in range(lod + 1):
            if self.temporal_w and i!=self.layer_count-1:
                w1 = F.interpolate(styles[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1))
                w2 = F.interpolate(styles[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1))
            else:
                w1 = styles[:, 2 * i + 0]
                w2 = styles[:, 2 * i + 1]
            x = self.decode_block[i](x, w1, w2, noise)
            if self.attention_block[i]:
                x = self.attention_block[i](x,w1) if self.attentional_style else self.attention_block[i](x)

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const

        for i in range(lod):
            if self.temporal_w and i!=self.layer_count-1:
                w1 = F.interpolate(styles[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1))
                w2 = F.interpolate(styles[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1))
            else:
                w1 = styles[:, 2 * i + 0]
                w2 = styles[:, 2 * i + 1]
            x = self.decode_block[i](x, w1, w2, noise)
            if self.attention_block[i]:
                x = self.attention_block[i](x,w1) if self.attentional_style else self.attention_block[i](x)
        x_prev = self.to_rgb[lod - 1](x)

        if self.temporal_w and lod!=self.layer_count-1:
            w1 = F.interpolate(styles[:, 2 * lod + 0],scale_factor=2**-(self.layer_count-lod-1))
            w2 = F.interpolate(styles[:, 2 * lod + 1],scale_factor=2**-(self.layer_count-lod-1))
        else:
            w1 = styles[:, 2 * lod + 0]
            w2 = styles[:, 2 * lod + 1]
        x = self.decode_block[lod](x, w1, w2, noise)
        if self.attention_block[lod]:
            x = self.attention_block[lod](x,w1) if self.attentional_style else self.attention_block[lod](x)
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, scale_factor = 2.0)
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
    def __init__(self, inputs, output, stride =1,lrmul=0.1,temporal_w=False,transpose=False,transform_kernel=False,use_sn=False):
        super(MappingBlock, self).__init__()
        if temporal_w:
            if transpose:
                self.map = sn(ln.ConvTranspose1d(inputs, output, 3,stride,1,0,lrmul=lrmul,transform_kernel=transform_kernel),use_sn=use_sn)
            else:
                self.map = sn(ln.Conv1d(inputs, output, 3,stride,1,lrmul=lrmul,transform_kernel=transform_kernel),use_sn=use_sn)
        else:
            self.map = sn(ln.Linear(inputs, output, lrmul=lrmul),use_sn=use_sn)

    def forward(self, x):
        x = F.leaky_relu(self.map(x), 0.2)
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
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256,temporal_w=False):
        super(VAEMappingToLatent_old, self).__init__()
        self.temporal_w = temporal_w
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            if not temporal_w:
                outputs = 2 * dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            else:
                outputs = mapping_fmaps
            block = MappingBlock(inputs, outputs, stride = 2 if i!=0 else 1,lrmul=0.1,temporal_w=temporal_w,transform_kernel=True if i!=0 else False)
            inputs = outputs
            self.map_blocks.append(block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))
        if temporal_w:
            self.Linear = sn(ln.Linear(inputs*8,2 * dlatent_size,lrmul=0.1))
    def forward(self, x):
        if x.dim()==3:
            x = torch.mean(x,dim=2)
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        if self.temporal_w:
            x = x.view(x.shape[0],x.shape[1]*x.shape[2])
            x = self.Linear(x)
        return x.view(x.shape[0], 2, x.shape[1] // 2)

@MAPPINGS.register("MappingToWord")
class MappingToWord(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, uniq_words=256, mapping_fmaps=256,temporal_w=False):
        super(MappingToWord, self).__init__()
        self.temporal_w = temporal_w
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            if not temporal_w:
                outputs = uniq_words if i == mapping_layers - 1 else mapping_fmaps
            else:
                outputs = mapping_fmaps
            block = MappingBlock(inputs, outputs , stride = 2 if i!=0 else 1,lrmul=0.1,temporal_w=temporal_w,transform_kernel=True if i!=0 else False)
            inputs = outputs
            self.map_blocks.append(block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))
        if temporal_w:
            self.Linear = sn(ln.Linear(inputs*8,uniq_words,lrmul=0.1))
    def forward(self, x):
        if x.dim()==3:
            x = torch.mean(x,dim=2)
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        if self.temporal_w:
            x = x.view(x.shape[0],x.shape[1]*x.shape[2])
            x = self.Linear(x)
        return x

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
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256,temporal_w=False):
        super(VAEMappingFromLatent, self).__init__()
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.temporal_w = temporal_w
        self.latent_size = latent_size
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        if temporal_w:
            self.Linear = sn(ln.Linear(dlatent_size,8*(latent_size//8)),use_sn=PARTIAL_SN)
            inputs = latent_size//8
        else:
            inputs = dlatent_size
        for i in range(mapping_layers):
            outputs = latent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, stride = i%2+1, lrmul=0.1,temporal_w=temporal_w,transform_kernel=True if i%2==1 else False, transpose=True,use_sn=PARTIAL_SN)
            inputs = outputs
            self.map_blocks.append(block)
            #print("dense %d %s" % ((i + 1), millify(count_parameters(block))))

    def forward(self, x):
        x = pixel_norm(x)
        if self.temporal_w:
            x = self.Linear(x)
            x = F.leaky_relu(x,0.2)
            x = x.view(x.shape[0],self.latent_size//8,8)
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        if self.temporal_w:
            return x.view(x.shape[0], 1, x.shape[1],x.shape[2]).repeat(1, self.num_layers, 1,1)
        else:
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
