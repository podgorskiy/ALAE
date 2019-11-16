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

import random
import losses
from net import *
import numpy as np


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="", encoder="", z_regression=False):
        super(Model, self).__init__()

        self.layer_count = layer_count
        self.z_regression = z_regression

        #self.mapping_tl = VAEMappingToLatent_old(
        self.mapping_tl = MAPPINGS["MappingToLatentNoStyle"](
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=1)

        #self.mapping_fl = VAEMappingFromLatent(
        self.mapping_fl = MAPPINGS["MappingFromLatent"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        #self.decoder = Generator(
        self.decoder = GENERATORS[generator]()

        #self.encoder = Encoder_old(
        self.encoder = ENCODERS[encoder]()

        self.decoder.apply(weights_init)
        self.encoder.apply(weights_init)
        self.mapping_tl.apply(weights_init)

        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

        self.criterion = nn.BCELoss()
        self.real_label = 1
        self.fake_label = 0


    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        w = self.mapping_fl(z)[:, 0]
        rec = self.decoder.forward(w)
        if return_styles:
            return w, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x)
        Z_ = self.mapping_tl(Z)
        return Z[:], torch.sigmoid(Z_[:, 0])

    def forward(self, x, lod, blend_factor, d_train, ae, alt):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=True, return_styles=True)

            Z, d_result_real = self.encode(rec, lod, blend_factor)

            assert Z.shape == s.shape

            if self.z_regression:
                Lae = torch.mean(((Z[:, 0] - z)**2))
            else:
                Lae = torch.mean(((Z - s.detach())**2)) * 10.0

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(x, lod, blend_factor)
            errD_real = self.criterion(d_result_real, torch.ones(x.shape[0]))

            _, d_result_fake = self.encode(Xp.detach(), lod, blend_factor)
            errD_fake = self.criterion(d_result_fake, torch.zeros(x.shape[0]))

            return errD_real + errD_fake
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)

            rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True)

            _, d_result_fake = self.encode(rec, lod, blend_factor)
            errG = self.criterion(d_result_fake, torch.ones(x.shape[0]))

            return errG

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
