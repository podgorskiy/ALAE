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

import random
import losses
from net import *
import numpy as np


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count,temporal_w=False,temporal_samples=128):
        super(DLatent, self).__init__()
        if temporal_w:
            buffer = torch.zeros(layer_count, dlatent_size, temporal_samples, dtype=torch.float32)
        else:
            buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)

class PPL_MEAN(nn.Module):
    def __init__(self):
        super(PPL_MEAN, self).__init__()
        buffer = torch.zeros(1, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, uniq_words=50, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="", encoder="", 
                 z_regression=False,average_w=False,spec_chans = 128,temporal_samples=128,temporal_w=False, global_w=True,temporal_global_cat = False,init_zeros=False,
                 residual=False,w_classifier=False,attention=None,cycle=None,w_weight=1.0,cycle_weight=1.0, attentional_style=False,heads=1,
                 ppl_weight=100,ppl_global_weight=100,ppld_weight=1,ppld_global_weight=1,common_z = False,
                 with_ecog = False, ecog_encoder="",suploss_on_ecog=False,less_temporal_feature=False):
        super(Model, self).__init__()

        self.layer_count = layer_count
        self.z_regression = z_regression
        self.common_z = common_z
        self.temporal_w = temporal_w
        self.global_w = global_w
        self.temporal_global_cat = temporal_global_cat
        self.w_classifier = w_classifier
        self.cycle = cycle
        self.w_weight=w_weight
        self.cycle_weight=cycle_weight
        self.ppl_weight = ppl_weight
        self.ppl_global_weight = ppl_global_weight
        self.ppld_weight = ppld_weight
        self.ppld_global_weight = ppld_global_weight
        self.suploss_on_ecog = suploss_on_ecog
        self.with_ecog = with_ecog
        latent_feature = latent_size//4 if (temporal_w and less_temporal_feature) else latent_size
        self.mapping_tl = MAPPINGS["MappingToLatent"](
            latent_size=latent_feature,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=5 if temporal_w else 3,
            temporal_w = temporal_w,
            global_w = global_w)

        self.mapping_tw = MAPPINGS["MappingToWord"](
            latent_size=latent_feature,
            uniq_words=uniq_words,
            mapping_fmaps=latent_size,
            mapping_layers=1,
            temporal_w = temporal_w)

        self.mapping_fl = MAPPINGS["MappingFromLatent"](
            num_layers=2 * layer_count,
            latent_size=latent_feature,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers,
            temporal_w = temporal_w,
            global_w = global_w)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_feature,
            channels=channels,
            spec_chans=spec_chans, temporal_samples = temporal_samples,
            temporal_w = temporal_w,
            global_w = global_w,
            temporal_global_cat = temporal_global_cat,
            init_zeros = init_zeros,
            residual = residual,
            attention=attention,
            attentional_style=attentional_style,
            heads = heads,
            )

        self.encoder = ENCODERS[encoder](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_feature,
            channels=channels,
            spec_chans=spec_chans, temporal_samples = temporal_samples,
            average_w=average_w,
            temporal_w = temporal_w,
            global_w = global_w,
            temporal_global_cat = temporal_global_cat,
            residual = residual,
            attention=attention,
            attentional_style=attentional_style,
            heads = heads,
            )

        if with_ecog:
            self.ecog_encoder = ECOG_ENCODER[ecog_encoder](
                latent_size = latent_feature,
                average_w = average_w,
                temporal_w=temporal_w,
                global_w = global_w,
                attention=attention,
                temporal_samples=temporal_samples,
                attentional_style=attentional_style,
                heads=heads,
            )

        self.dlatent_avg = DLatent(latent_feature, self.mapping_fl.num_layers,temporal_w=temporal_w)
        self.ppl_mean = PPL_MEAN()
        self.ppl_d_mean = PPL_MEAN()
        if temporal_w and global_w:
            self.dlatent_avg_global = DLatent(latent_feature, self.mapping_fl.num_layers,temporal_w=False)
            self.ppl_mean_global = PPL_MEAN()
            self.ppl_d_mean_global = PPL_MEAN()
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None, z_global=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False,ecog_only=True,ecog=None,mask_prior=None):
        if z is None:
            z = torch.randn(count, self.latent_size)
        if z_global is None:
            z_global = z if self.common_z else torch.randn(count, self.latent_size)
        if ecog is not None:
            styles_ecog = self.ecog_encoder(ecog,mask_prior)
            if self.temporal_w and self.global_w:
                styles_ecog, styles_ecog_global = styles_ecog
                s_ecog = styles_ecog.view(styles_ecog.shape[0], 1, styles_ecog.shape[1],styles_ecog.shape[2])
                styles_ecog = s_ecog.repeat(1, self.mapping_fl.num_layers, 1,1)
                s_ecog_global = styles_ecog_global.view(styles_ecog_global.shape[0], 1, styles_ecog_global.shape[1])
                styles_ecog_global = s_ecog_global.repeat(1, self.mapping_fl.num_layers, 1)
            else:
                if self.temporal_w:
                    s_ecog = styles_ecog.view(styles_ecog.shape[0], 1, styles_ecog.shape[1],styles_ecog.shape[2])
                    styles_ecog = s_ecog.repeat(1, self.mapping_fl.num_layers, 1,1)
                if self.global_w:
                    s_ecog = styles_ecog.view(styles_ecog.shape[0], 1, styles_ecog.shape[1])
                    styles_ecog = s_ecog.repeat(1, self.mapping_fl.num_layers, 1)
            if self.w_classifier:
                Z__ = self.mapping_tw(styles_ecog, styles_ecog_global)

        if (ecog is None) or (not ecog_only):
            if (self.temporal_w and self.global_w):
                styles = self.mapping_fl(z,z_global)
                styles, styles_global = styles
                styles = styles[:,0]
                styles_global = styles_global[:,0]
            else:
                styles = self.mapping_fl(z)[:, 0]

            if self.temporal_w and self.global_w:
                s = styles.view(styles.shape[0], 1, styles.shape[1],styles.shape[2])
                styles = s.repeat(1, self.mapping_fl.num_layers, 1,1)
                s_global = styles_global.view(styles_global.shape[0], 1, styles_global.shape[1])
                styles_global = s_global.repeat(1, self.mapping_fl.num_layers, 1)
            else:
                if self.temporal_w:
                    s = styles.view(styles.shape[0], 1, styles.shape[1],styles.shape[2])
                    styles = s.repeat(1, self.mapping_fl.num_layers, 1,1)
                if self.global_w:
                    s = styles.view(styles.shape[0], 1, styles.shape[1])
                    styles = s.repeat(1, self.mapping_fl.num_layers, 1)

            if self.dlatent_avg_beta is not None:
                with torch.no_grad():
                    batch_avg = styles.mean(dim=0)
                    self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)
                    if self.temporal_w and self.global_w:
                        batch_avg = styles_global.mean(dim=0)
                        self.dlatent_avg_global.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

            if mixing and self.style_mixing_prob is not None:
                if random.random() < self.style_mixing_prob:
                    cur_layers = (lod + 1) * 2
                    mixing_cutoff = random.randint(1, cur_layers)
                    layer_idx = torch.arange(self.mapping_fl.num_layers)
                    z2 = torch.randn(count, self.latent_size)
                    z2_global = z2 if self.common_z else torch.randn(count, self.latent_size)
                    if (self.temporal_w and self.global_w):
                        styles2 = self.mapping_fl(z2,z2_global)
                        styles2, styles2_global = styles2
                        styles2 = styles2[:,0]
                        styles2_global = styles2_global[:,0]
                    else:
                        styles2 = self.mapping_fl(z2)[:, 0]
                    if self.temporal_w and self.global_w:
                        styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1],styles2.shape[2]).repeat(1, self.mapping_fl.num_layers, 1,1)
                        styles = torch.where(layer_idx[np.newaxis, :, np.newaxis,np.newaxis] < mixing_cutoff, styles, styles2)
                        styles2_global = styles2_global.view(styles2_global.shape[0], 1, styles2_global.shape[1]).repeat(1, self.mapping_fl.num_layers, 1)
                        styles_global = torch.where(layer_idx[np.newaxis, :, np.newaxis] < mixing_cutoff, styles_global, styles2_global)
                    else:
                        if self.temporal_w:
                            styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1],styles2.shape[2]).repeat(1, self.mapping_fl.num_layers, 1,1)
                            styles = torch.where(layer_idx[np.newaxis, :, np.newaxis,np.newaxis] < mixing_cutoff, styles, styles2)
                        if self.global_w:
                            styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_fl.num_layers, 1)
                            styles = torch.where(layer_idx[np.newaxis, :, np.newaxis] < mixing_cutoff, styles, styles2)

            if (self.truncation_psi is not None) and not no_truncation:
                if self.temporal_w and self.global_w:
                    layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis,np.newaxis]
                    ones = torch.ones(layer_idx.shape, dtype=torch.float32)
                    coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
                    styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)
                    layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                    ones = torch.ones(layer_idx.shape, dtype=torch.float32)
                    coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
                    styles_global = torch.lerp(self.dlatent_avg_global.buff.data, styles_global, coefs)
                else:
                    if self.temporal_w:
                        layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis,np.newaxis]
                        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
                        coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
                        styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)
                    if self.global_w:
                        layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
                        coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
                        styles = torch.lerp(self.dlatent_avg_global.buff.data, styles, coefs)

        # import pdb; pdb.set_trace()
        if ecog is not None:
            if (not ecog_only):
                styles = torch.cat([styles_ecog,styles],dim=0)
                s = torch.cat([s_ecog,s],dim=0)
                if self.temporal_w and self.global_w:
                    styles_global = torch.cat([styles_ecog_global,styles_global],dim=0)
                    s_global = torch.cat([s_ecog_global,s_global],dim=0)
            else:
                styles = styles_ecog
                s = s_ecog
                if self.temporal_w and self.global_w:
                    styles_global = styles_ecog_global
                    s_global = s_ecog_global
                    
        if self.temporal_w and self.global_w:
            styles = (styles,styles_global)
        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        # import pdb; pdb.set_trace()
        if self.w_classifier:
            if return_styles:
                if self.temporal_w and self.global_w:
                    return (s, s_global), rec, Z__
                else:
                    return s, rec, Z__
            else:
                return rec,Z__
        else:
            if return_styles:
                if self.temporal_w and self.global_w:
                    return (s, s_global), rec
                else:
                    return s, rec
            else:
                return rec

    def encode(self, x, lod, blend_factor,word_classify=False):
        Z = self.encoder(x, lod, blend_factor)
        if self.temporal_w and self.global_w:
            Z,Z_global = Z
        
        Z_ = self.mapping_tl(Z[:,0],Z_global[:,0]) if (self.temporal_w and self.global_w) else self.mapping_tl(Z[:,0])
        if word_classify:
            Z__ = self.mapping_tw(Z[:,0],Z_global[:,0]) if (self.temporal_w and self.global_w) else self.mapping_tw(Z[:,0])
            if self.temporal_w and self.global_w:
                return (Z[:, :1],Z_global[:,:1]), Z_[:, 1, 0], Z__
            else:
                return Z[:, :1], Z_[:, 1, 0], Z__
        else:
            if self.temporal_w and self.global_w:
                return (Z[:, :1],Z_global[:,:1]), Z_[:, 1, 0]
            else:
                return Z[:, :1], Z_[:, 1, 0]

    def forward(self, x, lod, blend_factor, d_train, ae, tracker,words=None,apply_encoder_guide=False,apply_w_classifier=False,apply_cycle=True,apply_gp=True,apply_ppl=True,apply_ppl_d=False,ecog=None,sup=True,mask_prior=None,gan=True):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            if self.temporal_w and self.global_w:
                z_global = z if self.common_z else torch.randn(x.shape[0], self.latent_size)
            else:
                z_global = None
            s, rec = self.generate(lod, blend_factor, z=z, z_global=z_global, mixing=False, noise=True, return_styles=True,ecog=ecog,mask_prior=mask_prior)
            
            Z, _ = self.encode(rec, lod, blend_factor)
            do_cycle = self.cycle and apply_cycle
            if do_cycle:
                Z_real, _ = self.encode(x, lod, blend_factor)
                if self.temporal_w and self.global_w:
                    Z_real,Z_real_global = Z_real
                    Z_real_global = Z_real_global.repeat(1, self.mapping_fl.num_layers, 1)
                Z_real = Z_real.repeat(1, self.mapping_fl.num_layers, 1)
                rec = self.decoder.forward((Z_real,Z_real_global) if (self.temporal_w and self.global_w) else Z_real, lod, blend_factor, noise=True)
                Lcycle = self.cycle_weight*torch.mean((rec - x).abs())
                tracker.update(dict(Lcycle=Lcycle))
            else:
                Lcycle=0
                
            # assert Z.shape == s.shape

            if self.z_regression:
                Lae = self.w_weight*torch.mean(((Z[:, 0] - z)**2))
            else:
                if self.temporal_w and self.global_w:
                    Z,Z_global = Z
                    s,s_global = s
                    Lae = self.w_weight*(torch.mean(((Z - s.detach())**2)) + torch.mean(((Z_global - s_global.detach())**2)))
                else:
                    Lae = self.w_weight*torch.mean(((Z - s.detach())**2))
            tracker.update(dict(Lae=Lae))
            
            return Lae+Lcycle

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True,ecog=ecog,mask_prior=mask_prior)

            self.encoder.requires_grad_(True)
            
            if apply_w_classifier:
                _, d_result_real, word_logits = self.encode(x, lod, blend_factor,word_classify=True)
            else:
                xs = torch.cat([x,Xp.requires_grad_(True)],dim=0)
                w, d_result = self.encode(xs, lod, blend_factor)
                if self.temporal_w and self.global_w:
                    w, w_global = w
                    w_real_global = w_global[:w_global.shape[0]//2]
                    w_fake_global = w_global[w_global.shape[0]//2:]
                w_real = w[:w.shape[0]//2]
                w_fake = w[w.shape[0]//2:]
                d_result_real = d_result[:d_result.shape[0]//2]
                d_result_fake = d_result[d_result.shape[0]//2:]
                # w_real, d_result_real = self.encode(x, lod, blend_factor)
                # w_fake, d_result_fake = self.encode(Xp.requires_grad_(True), lod, blend_factor)

            loss_d = losses.critic_loss(d_result_fake, d_result_real)
            tracker.update(dict(loss_d=loss_d))
            if apply_gp:
                loss_gp = losses.discriminator_logistic_simple_gp(d_result_real, x)
                loss_d += loss_gp
            else:
                loss_gp=0
            if apply_ppl_d:
                path_loss_d, self.ppl_d_mean.buff.data, path_lengths_d = losses.pl_lengths_reg(xs, w, self.ppl_d_mean.buff.data,reg_on_gen=False,temporal_w = self.temporal_w)
                path_loss_d =path_loss_d*self.ppld_weight
                tracker.update(dict(path_loss_d=path_loss_d,path_lengths_d=path_lengths_d))
                if self.temporal_w and self.global_w and self.ppld_global_weight != 0:
                    path_loss_d_global, self.ppl_d_mean_global.buff.data, path_lengths_d_global = losses.pl_lengths_reg(xs, w_global, self.ppl_d_mean_global.buff.data,reg_on_gen=False,temporal_w = False)
                    path_loss_d_global = path_loss_d_global*self.ppld_global_weight
                    tracker.update(dict(path_loss_d_global=path_loss_d_global,path_lengths_d_global=path_lengths_d_global))
                    path_loss_d = path_loss_d+path_loss_d_global
                # path_loss_d =path_loss_d*self.ppl_weight
                # path_loss, self.ppl_mean.buff.data, path_lengths = losses.pl_lengths_reg(torch.cat([x,Xp],dim=0), torch.cat([w_real,w_fake],dim=0), self.ppl_mean.buff.data )
            else:
                path_loss_d=0
            if apply_w_classifier:
                loss_word = F.cross_entropy(word_logits,words)
                tracker.update(dict(loss_word=loss_word))
            else:
                loss_word=0
            return loss_d+loss_word+path_loss_d

        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)
                if self.temporal_w and self.global_w:
                    z_global = z if self.common_z else torch.randn(x.shape[0], self.latent_size)
                else:
                    z_global = None

            self.encoder.requires_grad_(False)
            s, rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), z_global=z_global, noise=True,return_styles=True,ecog=ecog,mask_prior=mask_prior)
            if self.temporal_w and self.global_w:
                s,s_global = s

            if gan:
                _, d_result_fake = self.encode(rec, lod, blend_factor)

                loss_g = losses.generator_logistic_non_saturating(d_result_fake)
                tracker.update(dict(loss_g=loss_g))
            else:
                loss_g = 0

            if apply_encoder_guide:
                Z_real, _ = self.encode(x, lod, blend_factor)
                if self.temporal_w and self.global_w:
                    Z_real,Z_real_global = Z_real
                    loss_w_sup = self.w_weight*(torch.mean(((Z_real - s)**2))+torch.mean(((Z_real_global - s_global)**2)))
                else:
                    loss_w_sup = self.w_weight*torch.mean(((Z_real - s)**2))
                tracker.update(dict(loss_w_sup=loss_w_sup))
            else:
                loss_w_sup=0

            if apply_ppl:
                path_loss, self.ppl_mean.buff.data, path_lengths = losses.pl_lengths_reg(s, rec, self.ppl_mean.buff.data,reg_on_gen=True,temporal_w = self.temporal_w)
                path_loss =path_loss*self.ppl_weight
                tracker.update(dict(path_loss=path_loss, path_lengths=path_lengths))
                if self.temporal_w and self.global_w:
                    path_loss_global, self.ppl_mean_global.buff.data, path_lengths_global = losses.pl_lengths_reg(s_global, rec, self.ppl_mean_global.buff.data,reg_on_gen=True,temporal_w = False)
                    path_loss_global =path_loss_global*self.ppl_global_weight
                    tracker.update(dict(path_loss_global=path_loss_global, path_lengths_global=path_lengths_global))
                    path_loss = path_loss+path_loss_global
            else:
                path_loss = 0
            if ecog is not None and sup:
                loss_sup = torch.mean((rec - x).abs())
                tracker.update(dict(loss_sup=loss_sup))
            else:
                loss_sup = 0
            if ecog is not None and self.suploss_on_ecog:
                return loss_g+ path_loss, loss_sup + loss_w_sup
            else:
                return loss_g+ path_loss+ loss_sup + loss_w_sup

    def lerp(self, other, betta,w_classifier=False):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters())+ list(self.mapping_fl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters()) + (list(other.dlatent_avg_global.parameters()) if (self.temporal_w and self.global_w) else []) + (list(self.mapping_tw.parameters()) if w_classifier else []) + (list(self.ecog_encoder.parameters()) if self.with_ecog else [])
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters()) + (list(other.dlatent_avg_global.parameters()) if (self.temporal_w and self.global_w) else [])  + (list(other.mapping_tw.parameters()) if w_classifier else []) + (list(other.ecog_encoder.parameters()) if self.with_ecog else [])
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)


class GenModel(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="", encoder="", z_regression=False):
        super(GenModel, self).__init__()

        self.layer_count = layer_count

        self.mapping_fl = MAPPINGS["MappingFromLatent"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_fl.num_layers,temporal_w=temporal_w)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None):
        styles = self.mapping_fl(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_fl.num_layers, 1)

        layer_idx = torch.arange(self.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
        styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, True)
        return rec

    def forward(self, x):
        return self.generate(self.layer_count-1, 1.0, z=x)
