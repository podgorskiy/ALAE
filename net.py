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


def style_mod(x, style1, style2=None, bias = True):
    if style1.dim()==2:
        style1 = style1.view(style1.shape[0], 2, x.shape[1], 1, 1)
    elif style1.dim()==3:
        style1 = style1.view(style1.shape[0], 2, x.shape[1], style1.shape[2], 1)
    if style2 is None:
        if bias:
            return torch.addcmul(style1[:, 1], value=1.0, tensor1=x, tensor2=style1[:, 0] + 1)
        else:
            return x*(style1[:,0]+1)
    else:
        if style2.dim()==2:
            style2 = style2.view(style2.shape[0], 2, x.shape[1], 1, 1)
        elif style2.dim()==3:
            style2 = style2.view(style2.shape[0], 2, x.shape[1], style2.shape[2], 1)
        if bias:
            return torch.addcmul(style1[:, 1]+style2[:, 1], value=1.0, tensor1=x, tensor2=(style1[:, 0] + 1)*(style2[:, 0] + 1))
        else:
            return x*(style1[:,0]+1)*(style2[:,0]+1)


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
    def __init__(self, latent_size,outputs,temporal_w=False,global_w=True,temporal_global_cat = False):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(outputs,affine=False, eps=1e-8)
        self.global_w = global_w
        self.temporal_w = temporal_w
        self.temporal_global_cat = temporal_global_cat and (temporal_w and global_w)
        if temporal_w and global_w:
            if self.temporal_global_cat:
                self.style = sn(ln.Conv1d(2*latent_size, 2 * outputs,1,1,0,gain=1))
            else:
                self.style = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0,gain=1))
                self.style_global = sn(ln.Linear(latent_size, 2 * outputs, gain=1))
        else:
            if temporal_w:
                self.style = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0,gain=1))
            if global_w:
                self.style = sn(ln.Linear(latent_size, 2 * outputs, gain=1))
    def forward(self,x,w=None,w_global=None):
        x = self.instance_norm(x)
        if self.temporal_w and self.global_w:
            if self.temporal_global_cat:
                w = torch.cat((w,w_global.unsqueeze(2).repeat(1,1,w.shape[2])),dim=1)
                x = style_mod(x,self.style(w))
            else:
                x = style_mod(x,self.style(w),self.style_global(w_global))
        else:
            x = style_mod(x,self.style(w))
        return x

class INencoder(nn.Module):
    def __init__(self, inputs,latent_size,temporal_w=False,global_w=True,temporal_global_cat = False,use_statistic=True):
        super(INencoder, self).__init__()
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.temporal_global_cat = temporal_global_cat and (temporal_w and global_w)
        self.use_statistic = use_statistic
        self.instance_norm = nn.InstanceNorm2d(inputs,affine=False)
        if global_w and not(temporal_w):
            self.style = sn(ln.Linear((2 * inputs) if use_statistic else inputs , latent_size))
        if temporal_w and not(global_w):
            self.style = sn(ln.Conv1d((2 * inputs) if use_statistic else inputs, latent_size,1,1,0))
        if temporal_w and global_w:
            if self.temporal_global_cat:
                self.style = sn(ln.Conv1d((4 * inputs) if use_statistic else inputs, 2*latent_size,1,1,0))
            else:
                self.style_local = sn(ln.Conv1d((2 * inputs) if use_statistic else inputs, latent_size,1,1,0))
                self.style_global = sn(ln.Linear((2 * inputs) if use_statistic else inputs, latent_size))

    def forward(self,x):
        m_local = torch.mean(x, dim=[3], keepdim=True)
        std_local = torch.sqrt(torch.mean((x - m_local) ** 2, dim=[3], keepdim=True)+1e-8)
        m_global = torch.mean(x, dim=[2,3], keepdim=True)
        std_global = torch.sqrt(torch.mean((x - m_global) ** 2, dim=[2,3], keepdim=True)+1e-8)
        if self.use_statistic:
            style_local = torch.cat((m_local,std_local),dim=1)
            style_global = torch.cat((m_global,std_global),dim=1)
        else:
            style_local = x
            style_global = x
        x = self.instance_norm(x)
        if self.global_w and not(self.temporal_w):
            w = self.style(style_global.view(style_global.shape[0], style_global.shape[1]))
            return x,w
        if self.temporal_w and not(self.global_w):
            w = self.style(style_local.view(style_local.shape[0], style_local.shape[1],style_local.shape[2]))
            return x,w
        if self.temporal_w and self.global_w:
            if self.temporal_global_cat:
                if self.use_statistic:
                    style = torch.cat((style_local,style_global.repeat(1,1,style_local.shape[2],1)),dim=1)
                else:
                    style = style_local
                w = self.style(style.view(style.shape[0], style.shape[1],style.shape[2]))
                w_local = w[:,:w.shape[1]//2]
                w_global = torch.mean(w[:,w.shape[1]//2:],dim=[2])
            else:
                w_local = self.style_local(style_local.view(style_local.shape[0], style_local.shape[1],style_local.shape[2]))
                if not self.use_statistic:
                    style_global = style_global.mean(dim=[2])
                w_global = self.style_global(style_global.view(style_global.shape[0], style_global.shape[1]))
            return x,w_local,w_global

class Attention(nn.Module):
  def __init__(self, inputs,temporal_w=False,global_w=True,temporal_global_cat = False,attentional_style=False,decoding=True,latent_size=None,heads=1,demod=False):
    super(Attention, self).__init__()
    # Channel multiplier
    self.inputs = inputs
    self.temporal_w = temporal_w
    self.global_w = global_w
    self.decoding = decoding
    self.attentional_style = attentional_style
    self.att_denorm = 8
    self.heads = heads
    self.demod = demod
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
            self.norm_theta = AdaIN(latent_size,inputs//self.att_denorm,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)
            self.norm_phi = AdaIN(latent_size,inputs//self.att_denorm,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)
            self.norm_g = AdaIN(latent_size,inputs//2,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)
        else:
            self.norm_theta = INencoder(inputs//self.att_denorm,latent_size,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)
            self.norm_phi = INencoder(inputs//self.att_denorm,latent_size,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)
            self.norm_g = INencoder(inputs//2,latent_size,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)

    if demod and attentional_style:
        self.theta = ln.StyleConv2d(inputs, inputs // self.att_denorm,kernel_size=1,latent_size=latent_size,stride=1,padding=0,
                                    bias=False, upsample=False,temporal_w=temporal_w,transform_kernel=False)
        self.phi = ln.StyleConv2d(inputs, inputs // self.att_denorm,kernel_size=1,latent_size=latent_size,stride=1,padding=0,
                                    bias=False, upsample=False,temporal_w=temporal_w,transform_kernel=False)
        self.g = ln.StyleConv2d(inputs, inputs // 2,kernel_size=1,latent_size=latent_size,stride=1,padding=0,
                                    bias=True, upsample=False,temporal_w=temporal_w,transform_kernel=False)
        self.o = ln.Conv2d(inputs // 2, inputs, 1,1,0, bias=True)


    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)

  def forward(self, x, w_local=None,w_global=None):
    # Apply convs
    x = x.contiguous()
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])    
    if w_local is not None and w_local.dim()==3:
        w_local_down = F.avg_pool1d(w_local, 2)    
    else:
        w_local_down = w_local
    if not self.demod:
        theta = self.norm_theta(theta,w_local,w_global) if (self.attentional_style and self.decoding) else self.norm_theta(theta)
        phi = self.norm_phi(phi,w_local_down,w_global) if (self.attentional_style and self.decoding) else self.norm_phi(phi)
        g = self.norm_g(g,w_local_down,w_global) if (self.attentional_style and self.decoding) else self.norm_g(g)
    if self.attentional_style and not self.decoding:
        if self.temporal_w and self.global_w:
            theta,w_theta_local,w_theta_global = theta
            phi,w_phi_local,w_phi_global = phi
            g,w_g_local,w_g_global = g
            w_phi_local = F.interpolate(w_phi_local,scale_factor=2,mode='linear')
            w_g_local = F.interpolate(w_g_local,scale_factor=2,mode='linear')
            w_local = w_theta_local+w_phi_local+w_g_local
            w_global = w_theta_global+w_phi_global+w_g_global
        else:
            theta,w_theta = theta
            phi,w_phi = phi
            g,w_g = g
            if w_phi.dim()==3:
                w_phi = F.interpolate(w_phi,scale_factor=2,mode='linear')
                w_g = F.interpolate(w_g,scale_factor=2,mode='linear')
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
    if (not self.attentional_style) or self.decoding:
        return self.gamma * o + x
    else:
        if self.temporal_w and self.global_w:
            return self.gamma * o + x, w_local, w_global
        else:
            return self.gamma * o + x, w

class ToWLatent(nn.Module):
    def __init__(self,inputs,latent_size,temporal_w=False,from_input=False):
        super(ToWLatent,self).__init__()
        self.temporal_w = temporal_w
        self.from_input = from_input
        if temporal_w:
            self.style = sn(ln.Conv1d(inputs, latent_size,1,1,0),use_sn=USE_SN)
        else:
            self.style = sn(ln.Linear(inputs, latent_size),use_sn=USE_SN)

    def forward(self, x):
        if not self.from_input:
            if self.temporal_w:
                m = torch.mean(x, dim=[3], keepdim=True)
                std = torch.sqrt(torch.mean((x - m) ** 2, dim=[3], keepdim=True)+1e-8)
            else:
                m = torch.mean(x, dim=[2, 3], keepdim=True)
                std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True)+1e-8)
        else:
            std = x
        if self.temporal_w:
            w = self.style(std.view(std.shape[0], std.shape[1],std.shape[2]))
        else:
            w = self.style(std.view(std.shape[0], std.shape[1]))
        return w

class ECoGMappingBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size,dilation=1,fused_scale=True,residual=False,resample=[]):
        super(ECoGMappingBlock, self).__init__()
        self.residual = residual
        self.inputs_resample = resample
        self.dim_missmatch = (inputs!=outputs)
        self.resample = resample
        if not self.resample:
            self.resample=1
        self.padding = list(np.array(dilation)*(np.array(kernel_size)-1)//2)
        # self.padding = [dilation[i]*(kernel_size[i]-1)//2 for i in range(len(dilation))]
        if residual:
            self.norm1 = nn.GroupNorm(min(inputs,32),inputs)
        else:
            self.norm1 = nn.GroupNorm(min(outputs,32),outputs)
        self.conv1 = sn(ln.Conv3d(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False))
        if self.inputs_resample or self.dim_missmatch:
            self.convskip = sn(ln.Conv3d(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False))
                
        self.conv2 = sn(ln.Conv3d(outputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False))
        self.norm2 = nn.GroupNorm(min(outputs,32),outputs)

    def forward(self,x):
        if self.residual:
            x = F.leaky_relu(self.norm1(x),0.2)
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
            else:
                x_skip = x
            x = F.leaky_relu(self.norm2(self.conv1(x)),0.2)
            x = self.conv2(x)
            x = x_skip + x
        else:
            x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
            x = F.leaky_relu(self.norm2(self.conv2(x)),0.2)
        return x



class DemodEncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False,fused_scale=True,temporal_w=False,temporal_samples=None,resample=False,spec_chans=None,attention=False,attentional_style=False,heads=1,channels=1):
        super(DemodEncodeBlock, self).__init__()
        self.last = last
        self.temporal_w = temporal_w
        self.attention = attention
        self.resample = resample
        self.fused_scale = False if temporal_w else fused_scale
        self.attentional_style = attentional_style
        self.fromrgb = FromRGB(channels, inputs,style = True,residual=False,temporal_w=temporal_w,latent_size=latent_size)
        self.conv1 = sn(ln.Conv2d(inputs, inputs, 3, 1, 1, bias=True))
        self.style1 = ToWLatent(inputs,latent_size,temporal_w=temporal_w)
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = Blur(inputs)
        if attention:
            self.non_local = Attention(inputs,temporal_w=temporal_w,attentional_style=attentional_style,decoding=False,latent_size=latent_size,heads=heads)
        if last:
            if self.temporal_w:
                self.conv_2 = sn(ln.Conv2d(inputs * spec_chans, outputs, 3, 1, 1, bias=True))
            else:
                self.dense = sn(ln.Linear(inputs * temporal_samples * spec_chans, outputs, bias = True))
        else:
            if resample and fused_scale:
                self.conv_2 = sn(ln.Conv2d(inputs, outputs, 3, 2, 1, bias=True, transform_kernel=True))
            else:
                self.conv_2 = sn(ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False))
        self.style2 = ToWLatent(outputs,latent_size,temporal_w=temporal_w,from_input=last)
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)

    def forward(self, spec,x):
        spec_feature,w0 = self.fromrgb(spec)
        x = (x+spec_feature) if (x is not None) else spec_feature
        if self.attention:
            x = self.non_local(x)
            if self.attentional_style:
                x,w_attn = x
        x = F.leaky_relu(self.conv1(x),0.2)
        w1 = self.style1(x)
        x = self.instance_norm_1(x)

        if self.last:
            if self.temporal_w:
                x = self.conv_2(x.view(x.shape[0], -1,x.shape[2]))
                x = F.leaky_relu(x, 0.2)
                w2 = self.style2(x.view(x.shape[0], x.shape[1],x.shape[2]))
            else:
                x = self.dense(x.view(x.shape[0], -1))
                x = F.leaky_relu(x, 0.2)
                w2 = self.style2(x.view(x.shape[0], x.shape[1]))

        else:
            x = F.leaky_relu(self.conv_2(self.blur(x)))
            if not self.fused_scale:
                x = downscale2d(x)
            w2 = self.style2(x)
            x = self.instance_norm_2(x)

        spec = F.avg_pool2d(spec,2,2)
        w = (w0+w1+w2+w_attn) if self.attentional_style else (w0+w1+w2)
        return spec,x,w

class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False,islast=False, fused_scale=True,temporal_w=False,global_w=True,temporal_global_cat = False,residual=False,resample=False,temporal_samples=None,spec_chans=None):
        super(EncodeBlock, self).__init__()
        self.conv_1 = sn(ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False))
        # self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        self.islast = islast
        self.fused_scale = False if temporal_w else fused_scale
        self.residual = residual
        self.resample=resample
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.temporal_global_cat = temporal_global_cat and (temporal_w and global_w)
        if last:
            if self.temporal_w:
                self.conv_2 = sn(ln.Conv2d(inputs * spec_chans, outputs, 3, 1, 1, bias=False))
            else:
                self.dense = sn(ln.Linear(inputs * temporal_samples * spec_chans, outputs))
        else:
            if resample and self.fused_scale:
                self.conv_2 = sn(ln.Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True))
            else:
                self.conv_2 = sn(ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False))

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.style_1 = INencoder(inputs,latent_size,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat,use_statistic=True)
        self.style_2 = INencoder(outputs,latent_size,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat,use_statistic=not(last))
        # self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)

        # if self.temporal_w:
        #     self.style_1 = sn(ln.Conv1d(2 * inputs, latent_size,1,1,0),use_sn=PARTIAL_SN)
        #     if last:
        #         self.style_2 = sn(ln.Conv1d(outputs, latent_size,1,1,0),use_sn=PARTIAL_SN)
        #     else:
        #         self.style_2 = sn(ln.Conv1d(2 * outputs, latent_size,1,1,0),use_sn=PARTIAL_SN)
        # else:
        #     self.style_1 = sn(ln.Linear(2 * inputs, latent_size),use_sn=PARTIAL_SN)
        #     if last:
        #         self.style_2 = sn(ln.Linear(outputs, latent_size),use_sn=PARTIAL_SN)
        #     else:
        #         self.style_2 = sn(ln.Linear(2 * outputs, latent_size),use_sn=PARTIAL_SN)

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
            x = F.leaky_relu(x,0.2)
            x_input = x
        x = self.conv_1(x) + self.bias_1
        if self.temporal_w and self.global_w:
            x,w1_local,w1_global = self.style_1(x)
        else:
            x,w1 = self.style_1(x)
        x = F.leaky_relu(x, 0.2)

        if self.last:
            if self.temporal_w:
                x = self.conv_2(x.view(x.shape[0], -1,x.shape[2]))
            else:
                x = self.dense(x.view(x.shape[0], -1))

            x = F.leaky_relu(x, 0.2)
            if self.temporal_w and self.global_w:
                x,w2_local,w2_global = self.style_2(x)
            else:
                x,w2 = self.style_2(x)
        else:
            x = self.conv_2(self.blur(x))
            x = x + self.bias_2

            if self.temporal_w and self.global_w:
                x,w2_local,w2_global = self.style_2(x)
            else:
                x,w2 = self.style_2(x)
            
            if not self.fused_scale:
                x = downscale2d(x)

        if not self.islast:
            if self.residual:
                x = self.skip(x_input)+x
        else:
            x = F.leaky_relu(x, 0.2)    
        
        if self.temporal_w and self.global_w:
            return x, w1_local, w1_global, w2_local, w2_global
        else:
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
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True, layer=0,temporal_w=False,global_w=True,temporal_global_cat = False,residual=False,resample = False):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.temporal_global_cat = temporal_global_cat and (temporal_w and global_w)
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
        # self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        # if temporal_w:
        #     self.style_1 = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0, gain=1),use_sn=PARTIAL_SN)
        #     self.style_2 = sn(ln.Conv1d(latent_size, 2 * outputs,1,1,0, gain=1),use_sn=PARTIAL_SN)
        # else: 
        #     self.style_1 = sn(ln.Linear(latent_size, 2 * outputs, gain=1),use_sn=PARTIAL_SN)
        #     self.style_2 = sn(ln.Linear(latent_size, 2 * outputs, gain=1),use_sn=PARTIAL_SN)
        self.style_1 = AdaIN(latent_size,outputs,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)
        self.style_2 = AdaIN(latent_size,outputs,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat)

        self.conv_2 = sn(ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False),use_sn=PARTIAL_SN)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        # self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        
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

    def forward(self, x, s1, s2, noise, s1_global=None, s2_global=None):
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

        # x = self.instance_norm_1(x)
        # x = style_mod(x, self.style_1(s1))
        if self.temporal_w and self.global_w:
            x = self.style_1(x,s1,s1_global)
        else:
            x = self.style_1(x,s1)

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
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8

        x = x + self.bias_2

        # x = self.instance_norm_2(x)
        # x = style_mod(x, self.style_2(s2))
        if self.temporal_w and self.global_w:
            x = self.style_2(x,s2,s2_global)
        else:
            x = self.style_2(x,s2)

        if self.residual:
            if self.has_first_conv:
                x = self.skip(x_input)+x
        else:
            x = F.leaky_relu(x, 0.2)

        return x

class DemodDecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale = True, layer=0,temporal_w=False,attention=False,attentional_style=False,heads=1,channels=1):
        super(DemodDecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.temporal_w = temporal_w
        self.attention = attention
        self.layer = layer
        if has_first_conv:
            if fused_scale:
                self.conv1 = ln.StyleConv2dtest(inputs, outputs, kernel_size=3, latent_size=latent_size, stride=2 , padding=1,
                                            bias=True,upsample=True,temporal_w=temporal_w,transform_kernel=True,transpose = True)
            else:
                self.conv1 = ln.StyleConv2dtest(inputs, outputs, kernel_size=3, latent_size=latent_size, stride=1 , padding=1,
                                            bias=True,upsample=True,temporal_w=temporal_w,transform_kernel=False,transpose = False)
            self.conv2 = ln.StyleConv2dtest(outputs, outputs, kernel_size=3, latent_size=latent_size, stride=1 , padding=1,
                                        bias=True,upsample=False,temporal_w=temporal_w,transform_kernel=False)
        else:
            self.conv1 = ln.StyleConv2dtest(inputs, outputs, kernel_size=3, latent_size=latent_size, stride=1 , padding=1,
                                    bias=True,upsample=False,temporal_w=temporal_w,transform_kernel=False)
        self.skip = ToRGB(outputs,channels,style=False,residual=False,temporal_w=temporal_w,latent_size=latent_size)
        # self.skip = ToRGB(outputs,channels,style=True,residual=False,temporal_w=temporal_w,latent_size=latent_size)
        if attention:
            self.att = Attention(outputs,temporal_w=temporal_w,attentional_style=attentional_style,decoding=True,latent_size=latent_size,heads=heads,demod=True)
        self.blur = Blur(channels)

    def forward(self,x,y,w,noise):
        x = F.leaky_relu(self.conv1(x,w,noise=noise),0.2)
        if not noise:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8
        if self.has_first_conv:
            x = F.leaky_relu(self.conv2(x,w,noise=noise),0.2)
            if not noise:
                x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8
        if self.attention:
            x = F.leaky_relu(self.att(x,w))
        skip = self.skip(x,w)
        if y is not None:
            y = upscale2d(y)
            y = self.blur(y)
            # y = F.interpolate(y,scale_factor=2,mode='bilinear')
        return (y+skip, x) if (y is not None) else (skip,x)


class FromECoG(nn.Module):
    def __init__(self, outputs,residual=False):
        super().__init__()
        self.residual=residual
        self.from_ecog = sn(ln.Conv3d(1, outputs, [9,1,1], 1, [4,0,0]))

    def forward(self, x):
        x = self.from_ecog(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)
        return x

class FromRGB(nn.Module):
    def __init__(self, channels, outputs,style = False,residual=False,temporal_w=False,latent_size=None):
        super(FromRGB, self).__init__()
        self.residual=residual
        self.from_rgb = sn(ln.Conv2d(channels, outputs, 1, 1, 0))
        self.style = style
        self.temporal_w = temporal_w
        if style:
            self.stylelayer = ToWLatent(outputs,latent_size,temporal_w=temporal_w)

    def forward(self, x):
        x = self.from_rgb(x)
        if self.style:
            w = self.stylelayer(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)

        return x if not self.style else (x, w)


class ToRGB(nn.Module):
    def __init__(self, inputs, channels,style = False,residual=False,temporal_w=False,latent_size=None):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.residual = residual
        self.style = style
        if style:
            self.to_rgb = ln.StyleConv2dtest(inputs, channels, kernel_size=1, latent_size=latent_size, stride=1 , padding=0, gain=0.03,
                                        bias=True,upsample=False,temporal_w=temporal_w,transform_kernel=False,demod=False)
        else:
            self.to_rgb = sn(ln.Conv2d(inputs, channels, 1, 1, 0, gain=0.03),use_sn=PARTIAL_SN)

    def forward(self, x,w=None):
        if self.residual:
            x = F.leaky_relu(x, 0.2)
        x = self.to_rgb(x,w) if (self.style and (w is not None) ) else self.to_rgb(x)
        return x

@ECOG_ENCODER.register("ECoGMappingDilation")
class ECoGMapping_Dilation(nn.Module):
    def __init__(self, latent_size,average_w = False,temporal_w=False,global_w=True,attention=None,temporal_samples=None,attentional_style=False,heads=1):
        super(ECoGMapping_Dilation, self).__init__()
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.from_ecog = FromECoG(16,residual=True)
        self.conv1 = ECoGMappingBlock(16,32,[5,1,1],residual=True,dilation=[2,1,1])
        self.conv2 = ECoGMappingBlock(32,64,[3,1,1],residual=True,dilation = [4,1,1])
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv3d(64,1,[3,1,1],1,[1,0,0])
        # self.mask = ln.Conv3d(64,1,[3,1,1],1,[4,0,0],dilation = [4,1,1])
        self.conv3 = ECoGMappingBlock(64,128,[3,3,3],residual=True,dilation = [8,2,2])
        self.conv4 = ECoGMappingBlock(128,256,[3,3,3],residual=True,dilation = [16,4,4])
        self.norm = nn.GroupNorm(32,256)
        self.conv5 = ln.Conv1d(256,256,3,1,16,dilation=16)
        if self.temporal_w:
            self.norm2 = nn.GroupNorm(32,256)
            self.conv6 = ln.Conv1d(256,256,3,1,1)
            self.norm3 = nn.GroupNorm(32,256)
            self.conv7 = ln.Conv1d(256,latent_size,3,1,1)
        if self.global_w:
            self.linear1 = ln.Linear(256*8,512)
            self.linear2 = ln.Linear(512,latent_size)
    def forward(self,ecog,mask_prior):
        x_all_global = []
        x_all_local = []
        for d in range(len(ecog)):
            x = ecog[d]
            x = x.reshape([-1,1,x.shape[1],15,15])
            mask_prior_d = mask_prior[d].reshape(-1,1,1,15,15)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
            mask = mask[:,:,8:-8]
            if mask_prior is not None:
                mask = mask*mask_prior_d
            x = x[:,:,8:-8]
            x = x*mask
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.max(-1)[0].max(-1)[0]
            x_common = self.conv5(F.leaky_relu(self.norm(x),0.2))
            if self.global_w:
                x_global = F.max_pool1d(x_common,16,16)
                x_global = x_global.flatten(1)
                x_global = self.linear1(F.leaky_relu(x_global,0.2))
                x_global = self.linear2(F.leaky_relu(x_global,0.2))
                x_global = F.leaky_relu(x_global,0.2)
                x_all_global += [x_global]
            if self.temporal_w:
                x_local = self.conv6(F.leaky_relu(self.norm2(x_common),0.2))
                x_local = self.conv7(F.leaky_relu(self.norm3(x_local),0.2))
                x_local = F.leaky_relu(x_local,0.2)
                x_all_local += [x_local]
        if self.global_w and self.temporal_w:
            x_all = (torch.cat(x_all_local,dim=0),torch.cat(x_all_global,dim=0))
        else:
            if self.temporal_w:
                x_all = torch.cat(x_all_local,dim=0)
            else:
                x_all = torch.cat(x_all_global,dim=0)
        return x_all

@ECOG_ENCODER.register("ECoGMappingBottleneck")
class ECoGMapping_Bottleneck(nn.Module):
    def __init__(self, latent_size,average_w = False,temporal_w=False,global_w=True,attention=None,temporal_samples=None,attentional_style=False,heads=1):
        super(ECoGMapping_Bottleneck, self).__init__()
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.from_ecog = FromECoG(16,residual=True)
        self.conv1 = ECoGMappingBlock(16,32,[5,1,1],residual=True,resample = [2,1,1])
        self.conv2 = ECoGMappingBlock(32,64,[3,1,1],residual=True,resample = [2,1,1])
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv3d(64,1,[3,1,1],1,[1,0,0])
        self.conv3 = ECoGMappingBlock(64,128,[3,3,3],residual=True,resample = [2,2,2])
        self.conv4 = ECoGMappingBlock(128,256,[3,3,3],residual=True,resample = [2,2,2])
        self.norm = nn.GroupNorm(32,256)
        self.conv5 = ln.Conv1d(256,256,3,1,1)
        if self.temporal_w:
            self.norm2 = nn.GroupNorm(32,256)
            self.conv6 = ln.ConvTranspose1d(256, 128, 3, 2, 1, transform_kernel=True)
            self.norm3 = nn.GroupNorm(32,128)
            self.conv7 = ln.ConvTranspose1d(128, 64, 3, 2, 1, transform_kernel=True)
            self.norm4 = nn.GroupNorm(32,64)
            self.conv8 = ln.ConvTranspose1d(64, 32, 3, 2, 1, transform_kernel=True)
            self.norm5 = nn.GroupNorm(32,32)
            self.conv9 = ln.ConvTranspose1d(32, latent_size, 3, 2, 1, transform_kernel=True)
        if self.global_w:
            self.linear1 = ln.Linear(256,128)
            self.linear2 = ln.Linear(128,latent_size)
    def forward(self,ecog,mask_prior):
        x_all_global = []
        x_all_local = []
        for d in range(len(ecog)):
            x = ecog[d]
            x = x.reshape([-1,1,x.shape[1],15,15])
            mask_prior_d = mask_prior[d].reshape(-1,1,1,15,15)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
            mask = mask[:,:,2:-2]
            if mask_prior is not None:
                mask = mask*mask_prior_d
            x = x[:,:,2:-2]
            x = x*mask
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.max(-1)[0].max(-1)[0]
            x_common = self.conv5(F.leaky_relu(self.norm(x),0.2))
            if self.global_w:
                x_global = x_common.max(-1)[0]
                x_global = self.linear1(F.leaky_relu(x_global,0.2))
                x_global = self.linear2(F.leaky_relu(x_global,0.2))
                x_global = F.leaky_relu(x_global,0.2)
                x_all_global += [x_global]
            if self.temporal_w:
                x_local = self.conv6(F.leaky_relu(self.norm2(x_common),0.2))
                x_local = self.conv7(F.leaky_relu(self.norm3(x_local),0.2))
                x_local = self.conv8(F.leaky_relu(self.norm4(x_local),0.2))
                x_local = self.conv9(F.leaky_relu(self.norm5(x_local),0.2))
                x_local = F.leaky_relu(x_local,0.2)
                x_all_local += [x_local]
        if self.global_w and self.temporal_w:
            x_all = (torch.cat(x_all_local,dim=0),torch.cat(x_all_global,dim=0))
        else:
            if self.temporal_w:
                x_all = torch.cat(x_all_local,dim=0)
            else:
                x_all = torch.cat(x_all_global,dim=0)
        return x_all

@ECOG_ENCODER.register("ECoGMappingDefault")
class ECoGMapping(nn.Module):
    def __init__(self, latent_size,average_w = False,temporal_w=False,global_w=True,attention=None,temporal_samples=None,attentional_style=False,heads=1):
        super(ECoGMapping, self).__init__()
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.from_ecog = FromECoG(16,residual=True)
        self.conv1 = ECoGMappingBlock(16,32,[5,1,1],residual=True,resample = [2,1,1])
        self.conv2 = ECoGMappingBlock(32,64,[3,1,1],residual=True,resample = [2,1,1])
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv3d(64,1,[3,1,1],1,[1,0,0])
        self.conv3 = ECoGMappingBlock(64,128,[3,3,3],residual=True,resample = [2,2,2])
        self.conv4 = ECoGMappingBlock(128,256,[3,3,3],residual=True,resample = [2,2,2])
        self.norm = nn.GroupNorm(32,256)
        self.conv5 = ln.Conv1d(256,256,3,1,1)
        if self.temporal_w:
            self.norm2 = nn.GroupNorm(32,256)
            self.conv6 = ln.Conv1d(256,256,3,1,1)
            self.norm3 = nn.GroupNorm(32,256)
            self.conv7 = ln.Conv1d(256,latent_size,3,1,1)
        if self.global_w:
            self.linear1 = ln.Linear(256*8,512)
            self.linear2 = ln.Linear(512,latent_size,gain=1)
    def forward(self,ecog,mask_prior):
        x_all_global = []
        x_all_local = []
        for d in range(len(ecog)):
            x = ecog[d]
            x = x.reshape([-1,1,x.shape[1],15,15])
            mask_prior_d = mask_prior[d].reshape(-1,1,1,15,15)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
            mask = mask[:,:,2:-2]
            if mask_prior is not None:
                mask = mask*mask_prior_d
            x = x[:,:,2:-2]
            x = x*mask
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.max(-1)[0].max(-1)[0]
            x_common = self.conv5(F.leaky_relu(self.norm(x),0.2))
            if self.global_w:
                x_global = x_common.flatten(1)
                x_global = self.linear1(F.leaky_relu(x_global,0.2))
                x_global = self.linear2(F.leaky_relu(x_global,0.2))
                x_all_global += [x_global]
            if self.temporal_w:
                x_local = self.conv6(F.leaky_relu(self.norm2(x_common),0.2))
                x_local = self.conv7(F.leaky_relu(self.norm3(x_local),0.2))
                x_all_local += [x_local]
        if self.global_w and self.temporal_w:
            x_all = (torch.cat(x_all_local,dim=0),torch.cat(x_all_global,dim=0))
        else:
            if self.temporal_w:
                x_all = torch.cat(x_all_local,dim=0)
            else:
                x_all = torch.cat(x_all_global,dim=0)
        return x_all
        


        
@ENCODERS.register("EncoderDemod")
class Encoder_Demod(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3,average_w = False,temporal_w=False,residual=False,attention=None,temporal_samples=None,spec_chans=None,attentional_style=False,heads=1):
        super(Encoder_Demod, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size
        self.average_w = average_w
        self.temporal_w = temporal_w
        self.attentional_style = attentional_style
        mul = 2
        inputs = startf
        self.encode_block = nn.ModuleList()
        self.attention_block = nn.ModuleList()
        resolution = 2 ** (self.layer_count + 1)
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)
            apply_attention = attention and attention[self.layer_count-i-1]
            current_spec_chans = spec_chans // 2**i
            current_temporal_samples = temporal_samples // 2**i
            last = i==(self.layer_count-1)
            fused_scale = resolution >= 128
            resolution //= 2
            block = DemodEncodeBlock(inputs, outputs, latent_size, last,temporal_w=temporal_w,fused_scale=fused_scale,resample=True,temporal_samples=current_temporal_samples,spec_chans=current_spec_chans,
                                    attention=apply_attention,attentional_style=attentional_style,heads=heads,channels=channels)
            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, spec, lod):
        if self.temporal_w:
            styles = torch.zeros(spec.shape[0], 1, self.latent_size,128)
        else:
            styles = torch.zeros(spec.shape[0], 1, self.latent_size)

        x = None
        for i in range(self.layer_count - lod - 1, self.layer_count):
            spec, x, w = self.encode_block[i](spec,x)
            if self.temporal_w and i!=0:
                w = F.interpolate(w,scale_factor=2**i)
            styles[:, 0] += w
        if self.average_w:
            styles /= (lod+1)
        return styles
    
    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


@ENCODERS.register("EncoderFormant")
class FormantEncoder(nn.Module):
   def __init__(self, n_mels=64, n_formants=4, k=30):
      super(FormantEncoder, self).__init__()
      self.n_mels = n_mels
      self.conv1 = ln.Conv1d(n_mels,64,3,1,1)
      self.norm1 = nn.GroupNorm(32,64)
      self.conv2 = ln.Conv1d(64,128,3,1,1)
      self.norm2 = nn.GroupNorm(32,128)

      self.conv_fundementals = ln.Conv1d(128,128,3,1,1)
      self.norm_fundementals = nn.GroupNorm(32,128)
      self.conv_f0 = ln.Conv1d(128,1,1,1,0)
      self.conv_amplitudes = ln.Conv1d(128,2,1,1,0)
      # self.conv_loudness = ln.Conv1d(128,1,1,1,0)

      self.conv_formants = ln.Conv1d(128,128,3,1,1)
      self.norm_formants = nn.GroupNorm(32,128)
      self.conv_formants_freqs = ln.Conv1d(128,n_formants,1,1,0)
      self.conv_formants_bandwidth = ln.Conv1d(128,n_formants,1,1,0)
      self.conv_formants_amplitude = ln.Conv1d(128,n_formants,1,1,0)

      self.amplifier = Parameter(torch.Tensor(1))
      with torch.no_grad():
         nn.init.constant_(self.amplifier,1.0)
   
   def forward(self,x):
      x = x.squeeze(dim=1).permute(0,2,1) #B * f * T
      loudness = torch.mean(x*0.5+0.5,dim=1,keepdim=True)
      loudness = F.softplus(self.amplifier)*loudness
      x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
      x_common = F.leaky_relu(self.norm2(self.conv2(x)),0.2)

      # loudness = F.relu(self.conv_loudness(x_common))
      amplitudes = F.softmax(self.conv_amplitudes(x_common),dim=1)

      x_fundementals = F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2)
      # f0 = F.sigmoid(self.conv_f0(x_fundementals))
      # f0 = F.tanh(self.conv_f0(x_fundementals)) * (16/64)*(self.n_mels/64) # 72hz < f0 < 446 hz
      f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
      # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
      # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice
      x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
      formants_freqs = F.sigmoid(self.conv_formants_freqs(x_formants))
      formants_freqs = torch.cumsum(formants_freqs,dim=1)
      formants_freqs = formants_freqs
      # formants_freqs = formants_freqs + f0
      formants_bandwidth = F.sigmoid(self.conv_formants_bandwidth(x_formants))
      formants_amplitude = F.softmax(self.conv_formants_amplitude(x_formants),dim=1)

      return f0,loudness,amplitudes,formants_freqs,formants_bandwidth,formants_amplitude


@ENCODERS.register("EncoderDefault")
class Encoder_old(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3,average_w = False,temporal_w=False,global_w=True,temporal_global_cat = False,residual=False,attention=None,temporal_samples=None,spec_chans=None,attentional_style=False,heads=1):
        super(Encoder_old, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size
        self.average_w = average_w
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.temporal_global_cat = temporal_global_cat
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
            non_local = Attention(inputs,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat,attentional_style=attentional_style,decoding=False,latent_size=latent_size,heads=heads) if apply_attention else None
            self.attention_block.append(non_local)
            fused_scale = resolution >= 128
            current_spec_chans = spec_chans // 2**i
            current_temporal_samples = temporal_samples // 2**i
            islast = i==(self.layer_count-1)
            block = EncodeBlock(inputs, outputs, latent_size, False, islast, fused_scale=fused_scale,temporal_w=temporal_w,global_w=global_w,temporal_global_cat=temporal_global_cat,residual=residual,resample=True,temporal_samples=current_temporal_samples,spec_chans=current_spec_chans)

            resolution //= 2

            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        if self.temporal_w and self.global_w:
            styles = torch.zeros(x.shape[0], 1, self.latent_size,128)
            styles_global = torch.zeros(x.shape[0], 1, self.latent_size)
        else:
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
                    if self.temporal_w and self.global_w:
                        x,s,s_global = x
                    else:
                        x,s = x
                    if self.temporal_w:
                        s = F.interpolate(s,scale_factor=2**i,mode='linear')
            if self.temporal_w and self.global_w:
                x, s1, s1_global, s2, s2_global = self.encode_block[i](x)
            else:
                x, s1, s2 = self.encode_block[i](x)
            if self.temporal_w and i!=0:
                s1 = F.interpolate(s1,scale_factor=2**i,mode='linear')
                s2 = F.interpolate(s2,scale_factor=2**i,mode='linear')
            if self.temporal_w and self.global_w:
                styles_global[:, 0] += s1_global + s2_global + (s_global if (self.attention_block[i] and self.attentional_style) else 0)
                styles[:, 0] += s1 + s2 + (s if (self.attention_block[i] and self.attentional_style) else 0)
            if self.temporal_w and self.global_w:
                styles_global[:, 0] += s1_global + s2_global + (s_global if (self.attention_block[i] and self.attentional_style) else 0)
        if self.average_w:
            styles /= (lod+1)
            if self.temporal_w and self.global_w:
                styles_global/=(lod+1)
                
        if self.temporal_w and self.global_w:
            return styles,styles_global
        else:
            return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        if self.temporal_w and self.global_w:
            styles = torch.zeros(x.shape[0], 1, self.latent_size,128)
            styles_global = torch.zeros(x.shape[0], 1, self.latent_size)
        else:
            if self.temporal_w:
                styles = torch.zeros(x.shape[0], 1, self.latent_size,128)
            else:
                styles = torch.zeros(x.shape[0], 1, self.latent_size)
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        if self.attention_block[self.layer_count - lod - 1]:
            x = self.attention_block[self.layer_count - lod - 1](x)
            if self.attentional_style:
                if self.temporal_w and self.global_w:
                    x,s,s_global = x
                else:
                    x,s = x
                if self.temporal_w:
                    s = F.interpolate(s,scale_factor=2**(self.layer_count - lod - 1),mode='linear')
        if self.temporal_w and self.global_w:
            x, s1, s1_global, s2, s2_global = self.encode_block[self.layer_count - lod - 1](x)
        else:
            x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        if self.temporal_w and (self.layer_count - lod - 1)!=0:
            s1 = F.interpolate(s1,scale_factor=2**(self.layer_count - lod - 1),mode='linear')
            s2 = F.interpolate(s2,scale_factor=2**(self.layer_count - lod - 1),mode='linear')
        styles[:, 0] += s1 * blend + s2 * blend + (s*blend if (self.attention_block[self.layer_count - lod - 1] and self.attentional_style) else 0)
        if self.temporal_w and self.global_w:
            styles_global[:, 0] += s1_global * blend + s2_global * blend + (s_global*blend if (self.attention_block[self.layer_count - lod - 1] and self.attentional_style) else 0)


        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            if self.attention_block[i]:
                x = self.attention_block[i](x)
                if self.attentional_style:
                    if self.temporal_w and self.global_w:
                        x,s,s_global = x
                    else:
                        x,s = x
                    if self.temporal_w:
                        s = F.interpolate(s,scale_factor=2**i,mode='linear')
            if self.temporal_w and self.global_w:
                x, s1, s1_global, s2, s2_global = self.encode_block[i](x)
            else:
                x, s1, s2 = self.encode_block[i](x)
            if self.temporal_w and i!=0:
                s1 = F.interpolate(s1,scale_factor=2**i,mode='linear')
                s2 = F.interpolate(s2,scale_factor=2**i,mode='linear')
            styles[:, 0] += s1 + s2 + (s if (self.attention_block[i] and self.attentional_style) else 0)
            if self.temporal_w and self.global_w:
                styles_global[:, 0] += s1_global + s2_global + (s_global if (self.attention_block[i] and self.attentional_style) else 0)
        if self.average_w:
            styles /= (lod+1)
            if self.temporal_w and self.global_w:
                styles_global/=(lod+1)
        if self.temporal_w and self.global_w:
            return styles,styles_global
        else:
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

@GENERATORS.register("GeneratorDemod")
class Generator_Demod(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3, temporal_samples=128,spec_chans=128,temporal_w=False,init_zeros=False,residual=False,attention=None,attentional_style=False,heads=1):
        super(Generator_Demod, self).__init__()
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
        self.initial_inputs = inputs
        self.init_specchans = spec_chans//2**(self.layer_count-1)
        self.init_temporalsamples = temporal_samples//2**(self.layer_count-1)
        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()
        self.attention_block = nn.ModuleList()
        self.decode_block: nn.ModuleList[DemodDecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128
            block = DemodDecodeBlock(inputs, outputs, latent_size, has_first_conv, layer=i,temporal_w=temporal_w,fused_scale=fused_scale,attention=attention and attention[i],attentional_style=attentional_style,heads=heads,channels=channels)

            resolution *= 2
            self.layer_to_resolution[i] = resolution
            self.decode_block.append(block)
            inputs = outputs
            mul //= 2
    
    def decode(self, styles, lod, noise):
        x = torch.randn([styles.shape[0], self.initial_inputs, self.init_temporalsamples, self.init_specchans])
        spec = None
        self.std_each_scale = []
        for i in range(lod + 1):
            if self.temporal_w and i!=self.layer_count-1:
                w1 = F.interpolate(styles[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                w2 = F.interpolate(styles[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1),mode='linear')
            else:
                w1 = styles[:, 2 * i + 0]
                w2 = styles[:, 2 * i + 1]
            spec, x = self.decode_block[i](x, spec, w1, noise)
            self.std_each_scale.append(spec.std())
        self.std_each_scale = torch.stack(self.std_each_scale)
        self.std_each_scale/=self.std_each_scale.sum()
        return spec

    def forward(self, styles, lod, blend, noise):
        if blend == 1:
            return self.decode(styles, lod, noise)
        else:
            return self.decode2(styles, lod, blend, noise)


@GENERATORS.register("GeneratorFormant")
class FormantSysth(nn.Module):
   def __init__(self, n_mels=64, k=30):
      super(FormantSysth, self).__init__()
      self.n_mels = n_mels
      self.k = k
      self.timbre = Parameter(torch.Tensor(1,1,n_mels))
      self.silient = -1
      with torch.no_grad():
         nn.init.constant_(self.timbre,1.0)
         # nn.init.constant_(self.silient,0.0)

   def formant_mask(self,freq,bandwith,amplitude):
      # freq, bandwith, amplitude: B*formants*time
      freq_cord = torch.arange(self.n_mels)
      time_cord = torch.arange(freq.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      freq = freq.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      bandwith = bandwith.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      amplitude = amplitude.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      masks = amplitude*torch.exp(-(grid_freq-freq)**2/(2*bandwith**2)) #B,time,freqchans, formants
      masks = masks.unsqueeze(dim=1) #B,1,time,freqchans
      return masks

   def mel_scale(self,hz):
      return (torch.log2(hz/440)+31/24)*24*self.n_mels/126
   
   def inverse_mel_scale(self,mel):
      return 440*2**(mel*126/24-31/24)

   def voicing(self,f0):
      #f0: B*1*time
      freq_cord = torch.arange(self.n_mels)
      time_cord = torch.arange(f0.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      f0 = f0.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, 1
      f0 = f0.repeat([1,1,1,self.k]) #B,time,1, self.k
      f0 = f0*(torch.arange(self.k)+1).reshape([1,1,1,self.k])
      bandwith = 24.7*(f0*4.37/1000+1)
      bandwith_lower = torch.clamp(f0-bandwith/2,min=0.001)
      bandwith_upper = f0+bandwith/2
      bandwith = self.mel_scale(bandwith_upper) - self.mel_scale(bandwith_lower)
      f0 = self.mel_scale(f0)
      # hamonics = torch.exp(-(grid_freq-f0)**2/(2*bandwith**2)) #gaussian
      hamonics = (1-((grid_freq-f0)/(3*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(3*bandwith)-0.5)*0.5+0.5) #welch
      # hamonics = torch.cos(np.pi*torch.abs(grid_freq-f0)/(4*bandwith))**2*(-torch.sign(torch.abs(grid_freq-f0)/(4*bandwith)-0.5)*0.5+0.5) #hanning
      hamonics = (hamonics.sum(dim=-1)*self.timbre).unsqueeze(dim=1) # B,1,T,F
      return hamonics

   def unvoicing(self,f0):
      return torch.ones([f0.shape[0],1,f0.shape[2],self.n_mels])

   def forward(self,f0,loudness,amplitudes,freq_formants,bandwidth_formants,amplitude_formants):
      # f0: B*1*T, amplitudes: B*2(voicing,unvoicing)*T, freq_formants,bandwidth_formants,amplitude_formants: B*formants*T
      amplitudes = amplitudes.unsqueeze(dim=-1)
      loudness = loudness.unsqueeze(dim=-1)
      f0_hz = self.inverse_mel_scale(f0)
      self.hamonics = self.voicing(f0_hz)
      self.noise = self.unvoicing(f0_hz)
      freq_formants = freq_formants*self.n_mels
      bandwidth_formants = bandwidth_formants*self.n_mels
      # excitation = amplitudes[:,0:1]*hamonics
      # excitation = loudness*(amplitudes[:,0:1]*hamonics)
      self.excitation = loudness*(amplitudes[:,0:1]*self.hamonics + amplitudes[:,-1:]*self.noise)
      self.mask = self.formant_mask(freq_formants,bandwidth_formants,amplitude_formants)
      self.mask_sum = self.mask.sum(dim=-1)
      speech = self.excitation*self.mask_sum + self.silient*torch.ones(self.mask_sum.shape)
      return speech


@GENERATORS.register("GeneratorDefault")
class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3, temporal_samples=128,spec_chans=128,temporal_w=False,global_w=True,temporal_global_cat = False,init_zeros=False,residual=False,attention=None,attentional_style=False,heads=1):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size
        self.temporal_w = temporal_w
        self.global_w=global_w
        self.temporal_global_cat = temporal_global_cat and (temporal_w and global_w)
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

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale, layer=i,temporal_w=temporal_w, global_w=global_w, temporal_global_cat=temporal_global_cat, residual=residual,resample=True)

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels,residual=residual))

            #print("decode_block%d %s styles in: %dl out resolution: %d" % (
            #    (i + 1), millify(count_parameters(block)), outputs, resolution))
            apply_attention = attention and attention[i]
            non_local = Attention(outputs,temporal_w=temporal_w, global_w=global_w, temporal_global_cat=temporal_global_cat, attentional_style=attentional_style,decoding=True,latent_size=latent_size,heads=heads) if apply_attention else None
            self.decode_block.append(block)
            self.attention_block.append(non_local)
            inputs = outputs
            mul //= 2

        self.to_rgb = to_rgb

    def decode(self, styles, lod, noise):
        if self.temporal_w and self.global_w:
            styles,styles_global = styles
        x = self.const
        self.std_each_scale = []
        for i in range(lod + 1):
            if self.temporal_w and i!=self.layer_count-1:
                w1 = F.interpolate(styles[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                w2 = F.interpolate(styles[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                # if self.temporal_w and self.global_w:
                #     w1_global = F.interpolate(styles_global[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                #     w2_global = F.interpolate(styles_global[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1),mode='linear')
            else:
                w1 = styles[:, 2 * i + 0]
                w2 = styles[:, 2 * i + 1]
            if self.temporal_w and self.global_w:
                w1_global = styles_global[:, 2 * i + 0]
                w2_global = styles_global[:, 2 * i + 1]
                x = self.decode_block[i](x, w1, w2, noise, w1_global, w2_global)
            else:
                x = self.decode_block[i](x, w1, w2, noise)
            if self.attention_block[i]:
                if self.temporal_w and self.global_w:
                    x = self.attention_block[i](x,w2, w2_global) if self.attentional_style else self.attention_block[i](x)
                else:
                    x = self.attention_block[i](x,w2) if self.attentional_style else self.attention_block[i](x)
            self.std_each_scale.append(x.std())
        self.std_each_scale = torch.stack(self.std_each_scale)
        self.std_each_scale/=self.std_each_scale.sum()

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise):
        if self.temporal_w and self.global_w:
            styles,styles_global = styles
        x = self.const

        for i in range(lod):
            if self.temporal_w and i!=self.layer_count-1:
                w1 = F.interpolate(styles[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                w2 = F.interpolate(styles[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                # if self.temporal_w and self.global_w:
                #     w1_global = F.interpolate(styles_global[:, 2 * i + 0],scale_factor=2**-(self.layer_count-i-1),mode='linear')
                #     w2_global = F.interpolate(styles_global[:, 2 * i + 1],scale_factor=2**-(self.layer_count-i-1),mode='linear')
            else:
                w1 = styles[:, 2 * i + 0]
                w2 = styles[:, 2 * i + 1]
            if self.temporal_w and self.global_w:
                w1_global = styles_global[:, 2 * i + 0]
                w2_global = styles_global[:, 2 * i + 1]
                x = self.decode_block[i](x, w1, w2, noise, w1_global, w2_global)
            else:
                x = self.decode_block[i](x, w1, w2, noise)
            if self.attention_block[i]:
                if self.temporal_w and self.global_w:
                    x = self.attention_block[i](x,w2,w2_global) if self.attentional_style else self.attention_block[i](x)
                else:
                    x = self.attention_block[i](x,w2) if self.attentional_style else self.attention_block[i](x)
        x_prev = self.to_rgb[lod - 1](x)

        if self.temporal_w and lod!=self.layer_count-1:
            w1 = F.interpolate(styles[:, 2 * lod + 0],scale_factor=2**-(self.layer_count-lod-1),mode='linear')
            w2 = F.interpolate(styles[:, 2 * lod + 1],scale_factor=2**-(self.layer_count-lod-1),mode='linear')
            if self.temporal_w and self.global_w:
                w1_global = F.interpolate(styles_global[:, 2 * lod + 0],scale_factor=2**-(self.layer_count-lod-1),mode='linear')
                w2_global = F.interpolate(styles_global[:, 2 * lod + 1],scale_factor=2**-(self.layer_count-lod-1),mode='linear')
        else:
            w1 = styles[:, 2 * lod + 0]
            w2 = styles[:, 2 * lod + 1]
            if self.temporal_w and self.global_w:
                w1_global = styles_global[:, 2 * lod + 0]
                w2_global = styles_global[:, 2 * lod + 1]
        if self.temporal_w and self.global_w:
            x = self.decode_block[lod](x, w1, w2, noise, w1_global, w2_global)
        else:
            x = self.decode_block[lod](x, w1, w2, noise)
        if self.attention_block[lod]:
            if self.temporal_w and self.global_w:
                x = self.attention_block[lod](x,w2,w2_global) if self.attentional_style else self.attention_block[lod](x)
            else:
                x = self.attention_block[lod](x,w2) if self.attentional_style else self.attention_block[lod](x)
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
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256,temporal_w=False, global_w=True):
        super(VAEMappingToLatent_old, self).__init__()
        self.temporal_w = temporal_w
        self.global_w = global_w
        inputs = 2* latent_size if (temporal_w and global_w) else latent_size
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
    def forward(self, x, x_global=None):
        # if x.dim()==3:
        #     x = torch.mean(x,dim=2)
        if (self.temporal_w and self.global_w):
            x = torch.cat((x,x_global.unsqueeze(2).repeat(1,1,x.shape[2])),dim=1)
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
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256,temporal_w=False,global_w = True):
        super(VAEMappingFromLatent, self).__init__()
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.latent_size = latent_size
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        if temporal_w and global_w:
            self.map_blocks_global: nn.ModuleList[MappingBlock] = nn.ModuleList()
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
        
        if temporal_w and global_w:
            inputs = dlatent_size
            for i in range(mapping_layers):
                outputs = latent_size if i == mapping_layers - 1 else mapping_fmaps
                block_global = MappingBlock(inputs, outputs, stride = i%2+1, lrmul=0.1,temporal_w=False,transform_kernel=True if i%2==1 else False, transpose=True,use_sn=PARTIAL_SN)
                inputs = outputs
                self.map_blocks_global.append(block_global)

    def forward(self, x,x_global=None):
        x = pixel_norm(x)
        if self.temporal_w:
            x = self.Linear(x)
            x = F.leaky_relu(x,0.2)
            x = x.view(x.shape[0],self.latent_size//8,8)
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        
        if self.temporal_w and self.global_w:
            x_global = pixel_norm(x_global)
            for i in range(self.mapping_layers):
                x_global = self.map_blocks_global[i](x_global)
            return x.view(x.shape[0], 1, x.shape[1],x.shape[2]).repeat(1, self.num_layers, 1,1), x_global.view(x_global.shape[0], 1, x_global.shape[1]).repeat(1, self.num_layers, 1)
        else:
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


