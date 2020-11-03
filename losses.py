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

import torch
import math
import torch.nn.functional as F


__all__ = ['kl', 'reconstruction', 'discriminator_logistic_simple_gp',
           'discriminator_gradient_penalty', 'generator_logistic_non_saturating']


def kl(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))


def reconstruction(recon_x, x, lod=None):
    return torch.mean((recon_x - x)**2)

def critic_loss(d_result_fake,d_result_real):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real)).mean()
    return loss

def discriminator_logistic_simple_gp(d_result_real, reals, r1_gamma=10.0):
    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def discriminator_gradient_penalty(d_result_real, reals, r1_gamma=10.0):
    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()


def pl_lengths_reg(inputs, outputs, mean_path_length, reg_on_gen, temporal_w=False,decay=0.01):
    # e.g. for generator, inputs = w (B x 1 x channel x T(optianal)), outputs=images (B x 1 x T x F)
    if reg_on_gen:
        num_pixels = outputs[0,0,0].numel() if temporal_w else outputs[0,0].numel() # freqbands if temporal else specsize
    else:
        num_pixels = outputs.shape[2] # latent space size per temporal sample
    pl_noise = torch.randn(outputs.shape).cuda() / math.sqrt(num_pixels)
    outputs = (outputs * pl_noise).sum()
    # if reg_on_gen:
    #     outputs = (outputs * pl_noise).sum(dim=[0,1,3]) if temporal_w else (outputs * pl_noise).sum()
    # else:
    #     outputs = (outputs * pl_noise).sum(dim=[0,1,2]) if temporal_w else (outputs * pl_noise).sum()

    pl_grads = torch.autograd.grad(outputs=outputs, inputs=inputs,
                          grad_outputs=torch.ones(outputs.shape).cuda(),
                          create_graph=True,retain_graph=True)[0]
    if reg_on_gen:          
        path_lengths = ((pl_grads ** 2).sum(dim=2).mean(dim=1)+1e-8).sqrt() #sum over feature, mean over repeated styles for each gen layers
    else:
        path_lengths = ((pl_grads ** 2).sum(dim=1)+1e-8).sqrt()
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()
    path_lengths = path_lengths.mean()
    return path_penalty,path_mean.detach(),path_lengths