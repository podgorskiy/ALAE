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
import random
import losses
from net import *
import numpy as np
from gradient_reversal import grad_reverse
import torch.nn.functional as F
import gradient_reversal


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3):
        super(Model, self).__init__()

        self.layer_count = layer_count

        self.mapping_tl = VAEMappingToLatent_old(
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_fl = VAEMappingFromLatent(
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = Generator(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = Encoder_old(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_fl(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_fl.num_layers, 1)

                layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x, lod, blend_factor)
        Z_ = self.mapping_tl(Z)
        return Z[:, :1], Z_[:, 1, 0]
        # Z, d = self.encoder(x, lod, blend_factor)
        # Z = self.mapping_tl(Z)
        # return Z, d

    def forward(self, x, lod, blend_factor, d_train, ae, alt):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=True, return_styles=True)

            Z, d_result_real = self.encode(rec, lod, blend_factor)

            assert Z.shape == s.shape

            Lae = torch.mean(((Z - s.detach())**2))

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(x, lod, blend_factor)

            _, d_result_fake = self.encode(Xp.detach(), lod, blend_factor)

            loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            return loss_d
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)

            rec = self.generate(lod, blend_factor, z=z.detach(), mixing=False, noise=True)

            _, d_result_fake = self.encode(rec, lod, blend_factor)

            loss_g = losses.generator_logistic_non_saturating(d_result_fake)

            return loss_g

        # LklZ = losses.kl(*Z)
        #
        # loss1 = LklZ * 0.02 + Lae
        #
        # Zr = self.encoder(grad_reverse(Xr), lod, blend_factor)
        #
        # Ladv = -losses.kl(*Zr) * alpha
        #
        # loss2 = Ladv * 0.02
        #
        # autoencoder_optimizer.zero_grad()
        # (loss1 + loss2).backward()
        # autoencoder_optimizer.step()


        # if d_train:
        #     with torch.no_grad():
        #         rec = self.generate(lod, blend_factor, count=x.shape[0])
        #     self.discriminator.requires_grad_(True)
        #     d_result_real = self.discriminator(x, lod, blend_factor).squeeze()
        #     d_result_fake = self.discriminator(rec.detach(), lod, blend_factor).squeeze()
        #
        #     loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
        #     return loss_d
        # else:
        #     rec = self.generate(lod, blend_factor, count=x.shape[0])
        #     self.discriminator.requires_grad_(False)
        #     d_result_fake = self.discriminator(rec, lod, blend_factor).squeeze()
        #     loss_g = losses.generator_logistic_non_saturating(d_result_fake)
        #     return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
