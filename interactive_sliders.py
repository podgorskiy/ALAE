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
from skimage.transform import resize
import utils

from PIL import Image
import bimpy


lreq.use_implicit_lreq.set(True)


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
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    path = 'realign1024_2_v'
    #path = 'imagenet256x256'
    # path = 'realign128x128'

    paths = list(os.listdir(path))

    paths = ['00096.png', '00002.png', '00106.png', '00103.png', '00013.png', '00037.png']#sorted(paths)
    #random.seed(3456)
    #random.shuffle(paths)

    ctx = bimpy.Context()
    v0 = bimpy.Float(0)
    v1 = bimpy.Float(0)
    v2 = bimpy.Float(0)
    v3 = bimpy.Float(0)
    v4 = bimpy.Float(0)
    v10 = bimpy.Float(0)
    v11 = bimpy.Float(0)
    v17 = bimpy.Float(0)
    v19 = bimpy.Float(0)

    w0 = torch.tensor(np.load("direction_%d.npy" % 0), dtype=torch.float32)
    w1 = torch.tensor(np.load("direction_%d.npy" % 1), dtype=torch.float32)
    w2 = torch.tensor(np.load("direction_%d.npy" % 2), dtype=torch.float32)
    w3 = torch.tensor(np.load("direction_%d.npy" % 3), dtype=torch.float32)
    w4 = torch.tensor(np.load("direction_%d.npy" % 4), dtype=torch.float32)
    w10 = torch.tensor(np.load("direction_%d.npy" % 10), dtype=torch.float32)
    w11 = torch.tensor(np.load("direction_%d.npy" % 11), dtype=torch.float32)
    w17 = torch.tensor(np.load("direction_%d.npy" % 17), dtype=torch.float32)
    w19 = torch.tensor(np.load("direction_%d.npy" % 19), dtype=torch.float32)

    _latents = None

    def loadNext():
        img = np.asarray(Image.open(path + '/' + paths[0]))
        img_src = img
        paths.pop(0)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        _latents = encode(x[None, ...].cuda())
        latents = _latents[0, 0]

        latents -= model.dlatent_avg.buff.data[0]

        v0.value = (latents * w0).sum()
        v1.value = (latents * w1).sum()
        v2.value = (latents * w2).sum()
        v3.value = (latents * w3).sum()
        v4.value = (latents * w4).sum()
        v10.value = (latents * w10).sum()
        v11.value = (latents * w11).sum()
        v17.value = (latents * w17).sum()
        v19.value = (latents * w19).sum()

        latents = latents - v0.value * w0
        latents = latents - v1.value * w1
        latents = latents - v2.value * w2
        latents = latents - v3.value * w3
        latents = latents - v10.value * w10
        latents = latents - v11.value * w11
        latents = latents - v17.value * w17
        latents = latents - v19.value * w19
        return latents, _latents, img_src

    latents, _latents, img_src = loadNext()

    ctx.init(1800, 1600, "Styles")

    def update_image(w, _w):
        with torch.no_grad():
            w = w + model.dlatent_avg.buff.data[0]
            w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)

            layer_idx = torch.arange(model.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = (7 + 1) * 2
            mixing_cutoff = cur_layers
            styles = torch.where(layer_idx < mixing_cutoff, w, _latents[0])

            x_rec = decode(styles)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    im = update_image(latents, _latents)
    print(im.shape)
    im = bimpy.Image(im)

    display_original = True

    while(not ctx.should_close()):
        with ctx:
            W = latents + w0 * v0.value + w1 * v1.value + w2 * v2.value + w3 * v3.value + w4 * v4.value + w10 * v10.value + w11 * v11.value + w17 * v17.value + w19 * v19.value

            if display_original:
                im = bimpy.Image(img_src)
            else:
                im = bimpy.Image(update_image(W, _latents))

            # if bimpy.button('Ok'):
            bimpy.image(im)
            bimpy.begin("Controls")
            bimpy.slider_float("female <-> male", v0, -30.0, 30.0)
            bimpy.slider_float("smile", v1, -30.0, 30.0)
            bimpy.slider_float("attractive", v2, -30.0, 30.0)
            bimpy.slider_float("wavy-hair", v3, -30.0, 30.0)
            bimpy.slider_float("young", v4, -30.0, 30.0)
            bimpy.slider_float("big lips", v10, -30.0, 30.0)
            bimpy.slider_float("big nose", v11, -30.0, 30.0)
            bimpy.slider_float("chubby", v17, -30.0, 30.0)
            bimpy.slider_float("glasses", v19, -30.0, 30.0)

            if bimpy.button('Next'):
                latents, _latents, img_src = loadNext()
                display_original = True
            if bimpy.button('Display Reconstruction'):
                display_original = False
            bimpy.end()

    exit()


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_ffhq_z.yaml',
        world_size=gpu_count, write_log=False)
