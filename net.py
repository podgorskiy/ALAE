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


class VAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(VAE, self).__init__()
        self.maxf = 256

        d = 64
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        self.from_rgb = nn.Conv2d(3, d, 1, 1, 0)
        self.to_rgb = nn.Conv2d(d, 3, 1, 1, 0)

        mul = 2
        inputs = d
        for i in range(self.layer_count):
            outputs = min(self.maxf, d * mul)
            setattr(self, "conv%d_1" % (i + 1), nn.Conv2d(inputs, inputs, 3, 1, 1))
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, outputs, 3, 2, 1))
            setattr(self, "conv%d_1_bn" % (i + 1), nn.InstanceNorm2d(inputs, affine=True))
            setattr(self, "conv%d_bn" % (i + 1), nn.InstanceNorm2d(outputs, affine=True))
            inputs = outputs
            mul *= 2

        self.d_max = inputs

        #self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        #self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

        #self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul //= 4

        for i in range(self.layer_count):
            outputs = min(self.maxf, d * mul)
            setattr(self, "deconv%d_1" % (i + 1), nn.ConvTranspose2d(inputs, outputs, 3, 2, 1, output_padding=1))
            setattr(self, "deconv%d" % (i + 1), nn.Conv2d(outputs, outputs, 3, 1, 1))
            setattr(self, "deconv%d_1_bn" % (i + 1), nn.InstanceNorm2d(outputs, affine=True))
            setattr(self, "deconv%d_bn" % (i + 1), nn.InstanceNorm2d(outputs, affine=True))
            inputs = outputs
            mul //= 2

        self.const = Parameter(torch.Tensor(1, self.d_max, 4, 4))
        init.normal_(self.const)

    def encode(self, x):
        styles = []

        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count):
            x = getattr(self, "conv%d_1" % (i + 1))(x)
            x = F.leaky_relu(x, 0.2)

            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m)**2, dim=[2, 3], keepdim=True))
            styles.append((m, std))

            x = getattr(self, "conv%d_1_bn" % (i + 1))(x)
            #print("Encode. Layer %d 1" %(i), x.shape)

            ##########################################################

            x = getattr(self, "conv%d" % (i + 1))(x)
            x = F.leaky_relu(x, 0.2)

            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m)**2, dim=[2, 3], keepdim=True))
            styles.append((m, std))

            x = getattr(self, "conv%d_bn" % (i + 1))(x)
            #print("Encode. Layer %d 2" %(i), x.shape)

        return styles

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, styles):
        x = self.const

        styles = styles[:]

        for i in range(self.layer_count):
            s = styles.pop()
            #print("Decode. Style ", s[0].shape, "Data", x.shape)
            x = x * s[1] + s[0]
            x = getattr(self, "deconv%d_1" % (i + 1))(x)
            x = F.leaky_relu(x, 0.2)

            x = getattr(self, "deconv%d_1_bn" % (i + 1))(x)

            s = styles.pop()
            #print("Decode. Style ", s[0].shape, "Data", x.shape)
            x = x * s[1] + s[0]
            x = getattr(self, "deconv%d" % (i + 1))(x)
            x = F.leaky_relu(x, 0.2)

            x = getattr(self, "deconv%d_bn" % (i + 1))(x)

        x = self.to_rgb(x)
        return x

    def forward(self, x):
        styles = self.encode(x)
        return self.decode(styles)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
