# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Perceptual Path Length (PPL)."""

import numpy as np
import torch
import pickle
from net import *
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from model_z_gan import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from dataloader import *
import scipy.linalg

from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import argparse
import logging
import sys
import lreq
from skimage.transform import resize
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt
import utils
import PIL


def draw_uncurated_result_figure(cfg, png, model, cx, cy, cw, ch, rows, lods, seed):
    print(png)
    N = sum(rows * 2**lod for lod in lods)
    images = []

    rnd = np.random.RandomState(3456)
    for i in range(N):
        latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
        samplez = torch.tensor(latents).float().cuda()
        image = model.generate(cfg.DATASET.MAX_RESOLUTION_LEVEL-2, 1, samplez, 1, mixing=False)
        images.append(image[0])

    canvas = PIL.Image.new('RGB', (sum(cw // 2**lod for lod in lods), ch * rows), 'white')
    image_iter = iter(list(images))
    for col, lod in enumerate(lods):
        for row in range(rows * 2**lod):
            image = PIL.Image.fromarray(next(image_iter).cpu().numpy().transpose(2, 0, 1), 'RGB')
            image = image.crop((cx, cy, cx + cw, cy + ch))
            image = image.resize((cw // 2**lod, ch // 2**lod), PIL.Image.ANTIALIAS)
            canvas.paste(image, (sum(cw // 2**lod for lod in lods[:col]), row * ch // 2**lod))
    canvas.save(png)


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

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Evaluating FID metric")

    decoder = nn.DataParallel(decoder)

    with torch.no_grad():
        draw_uncurated_result_figure(cfg, 'figure02-uncurated-ffhq.png', model, cx=0, cy=0, cw=1024, ch=1024, rows=3, lods=[0,1,2,2,3,3], seed=5)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_celeba.yaml',
        world_size=gpu_count, write_log=False)
