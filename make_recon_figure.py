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

from PIL import Image


lreq.use_implicit_lreq.set(True)

im_size = 128


def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


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
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, 1.2 * ones, ones)
        x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    rnd = np.random.RandomState(4)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)

    im_size = 128

    # path = 'realign1024x1024'
    path = 'realign128x128'

    src = []
    for filename in os.listdir(path):
        img = np.asarray(Image.open(path + '/' + filename))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        src.append(x)

    with torch.no_grad():
        reconstructions = []
        for s in src:
            latents = encode(s[None, ...])
            reconstructions.append(decode(latents))

    src *= 4
    reconstructions *= 4

    initial_resolution = 512

    lods_down = 2
    padding_step = 4

    width = 0
    height = 0

    current_padding = 0

    final_resolution = initial_resolution
    for _ in range(lods_down):
        final_resolution /= 2

    for i in range(lods_down + 1):
        width += current_padding * 2 ** (lods_down - i)
        height += current_padding * 2 ** (lods_down - i)
        current_padding += padding_step

    width += 2 ** (lods_down + 1) * final_resolution
    height += (lods_down + 1) * initial_resolution

    width = int(width)
    height = int(height)

    def make_part(current_padding):
        canvas = np.ones([3, height + current_padding * 2 ** (lods_down + 1), width])

        padd = 0

        initial_padding = current_padding

        height_padding = 0

        for i in range(lods_down + 1):
            for x in range(2 ** i):
                for y in range(2 ** i):
                    try:
                        ims = src.pop()
                        imr = reconstructions.pop()[0]
                        ims = ims.cpu().detach().numpy()
                        imr = imr.cpu().detach().numpy()

                        res = int(initial_resolution / 2 ** i)

                        ims = resize(ims, (3, initial_resolution / 2 ** i, initial_resolution / 2 ** i))
                        imr = resize(imr, (3, initial_resolution / 2 ** i, initial_resolution / 2 ** i))

                        place(canvas, ims,
                              current_padding + x * (2 * res + current_padding),
                              i * initial_resolution + height_padding + y * (res + current_padding))

                        place(canvas, imr,
                              current_padding + res + x * (2 * res + current_padding),
                              i * initial_resolution + height_padding + y * (res + current_padding))

                    except IndexError:
                        return canvas

            height_padding += initial_padding * 2

            current_padding -= padding_step
            padd += padding_step
        return canvas

    canvas = [make_part(current_padding), make_part(current_padding), make_part(current_padding), make_part(current_padding)]

    canvas = np.concatenate(canvas, axis=2)

    save_image(torch.Tensor(canvas), 'reconstructions_multiresolution.png')


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_z.yaml',
        world_size=gpu_count, write_log=False)
