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

from __future__ import print_function
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
from model import Model
from net import *
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from model_z_gan import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver


from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import argparse
import logging
import sys
import bimpy
import lreq

from PIL import Image


lreq.use_implicit_lreq.set(True)

im_size = 128


def place(canvas, image, x, y):
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size : (y + 1) * im_size, x * im_size : (x + 1) * im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        layer_count = cfg.MODEL.LAYER_COUNT
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        # layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
        # ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        # coefs = torch.where(layer_idx < model.truncation_cutoff, 1.2 * ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        #
        return model.decoder(x, layer_count - 1, 1, noise=True)

    rnd = np.random.RandomState(4)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)

    im_size = 128

    # with open('data_selected.pkl', 'rb') as pkl:
    #     data_train = pickle.load(pkl)
    #
    #     def process_batch(batch):
    #         data = [x.transpose((2, 0, 1)) for x in batch]
    #         x = torch.tensor(np.asarray(data, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
    #         return x
    #     data_train = process_batch(data_train[:32])

    #    Bold man with glasses
    #    Blond yang woman
    #    Kid
    #    Woman with pose different than frontal and smiling
    #    Older man smiling

    #    Dark hair smiling yang wonam
    #    Blond smiling yang woman with non frontal pose
    #    Black young woman
    #    Brown color (Asian?) girl
    #    Yang man hairy
    #   Smiling brown hair woman

    # src_ids = [18, 2, 10, 1, 13]
    # dst_ids = [3, 4, 14, 21, 20, 19]
    # src_originals = torch.stack([data_train[x] for x in src_ids])
    # dst_originals = torch.stack([data_train[x] for x in dst_ids])

    path = 'test_images/set_4/'
    src_len = 5
    dst_len = 6

    src_originals = []
    for i in range(src_len):
        im = np.asarray(Image.open(path + 'src/%d.png' % i))
        im = im.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        src_originals.append(x)
    src_originals = torch.stack([x for x in src_originals])
    dst_originals = []
    for i in range(dst_len):
        im = np.asarray(Image.open(path + 'dst/%d.png' % i))
        im = im.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        dst_originals.append(x)
    dst_originals = torch.stack([x for x in dst_originals])

    src_latents = encode(src_originals)
    src_images = decode(src_latents)

    dst_latents = encode(dst_originals)
    dst_images = decode(dst_latents)

    canvas = np.zeros([3, im_size * (dst_len + 2), im_size * (src_len + 2)])

    for i in range(src_len):
        save_image(src_originals[i] * 0.5 + 0.5, 'source_%d.jpg' % i)
        #save_image(src_originals[i] * 0.5 + 0.5, path + 'src/%d.png' % i)
        place(canvas, src_originals[i], 2 + i, 0)
        place(canvas, src_images[i], 2 + i, 1)

    for i in range(dst_len):
        save_image(dst_originals[i] * 0.5 + 0.5, 'dst_coarse_%d.jpg' % i)
        #save_image(dst_originals[i] * 0.5 + 0.5, path + 'dst/%d.png' % i)
        place(canvas, dst_originals[i], 0, 2 + i)
        place(canvas, dst_images[i], 1, 2 + i)

    style_ranges = [range(0, 4)] * 6 + [range(4, 8)] * 2 + [range(8, layer_count * 2)]

    def mix_styles(style_src, style_dst, r):
        style = style_dst.clone()
        style[:, r] = style_src[:, r]
        return style

    for row in range(dst_len):
        row_latents = torch.stack([dst_latents[row]] * src_len)
        style = mix_styles(src_latents, row_latents, style_ranges[row])
        rec = model.decoder(style, layer_count - 1, 1, noise=True)
        for j in range(rec.shape[0]):
            save_image(rec[j] * 0.5 + 0.5, 'rec_coarse_%d_%d.jpg' % (row, j))
            place(canvas, rec[j], 2 + j, 2 + row)

    # cut_layer_b = len(styless) - 1 - 4
    # cut_layer_e = len(styless) - 1 - 10

    # for i in [3, 4]:
    #     for j in range(im_count):
    #         style_a = [x[i].unsqueeze(0) for x in stylesd[:cut_layer_b]]
    #         style_b = [x[j].unsqueeze(0) for x in styless[cut_layer_b:cut_layer_e]]
    #         style_c = [x[i].unsqueeze(0) for x in stylesd[cut_layer_e:]]
    #         style = style_a + style_b + style_c
    #
    #         save_image(xd[i] * 0.5 + 0.5, 'dst_mid_%d.jpg' % i)
    #
    #         rec = vae.decode(style, layer_count - 1, True)
    #         save_image(rec * 0.5 + 0.5, 'rec_mid_%d_%d.jpg' % (i, j))
    #         # place(canvas, rec[0], 2 + i, 2 + j)
    #
    # cut_layer_b = len(styless) - 1 - 0
    # cut_layer_e = len(styless) - 1 - 4
    #
    # for i in [1]:
    #     for j in range(im_count):
    #         style_a = [x[i].unsqueeze(0) for x in stylesd[:cut_layer_b]]
    #         style_b = [x[j].unsqueeze(0) for x in styless[cut_layer_b:cut_layer_e]]
    #         style_c = [x[i].unsqueeze(0) for x in stylesd[cut_layer_e:]]
    #         style = style_a + style_b + style_c
    #
    #         save_image(xd[i] * 0.5 + 0.5, 'dst_fine_%d.jpg' % i)
    #
    #         rec = vae.decode(style, layer_count - 1, True)
    #         save_image(rec * 0.5 + 0.5, 'rec_fine_%d.jpg' % j)
    #         # place(canvas, rec[0], 2 + i, 2 + j)

    save_image(torch.Tensor(canvas), 'reconstruction.png')


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_z.yaml',
        world_size=gpu_count, write_log=False)
