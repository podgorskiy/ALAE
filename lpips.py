# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Perceptual Path Length (PPL)."""

import numpy as np
import tensorflow as tf
import torch
import dnnlib
import dnnlib.tflib
import dnnlib.tflib as tflib
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

dnnlib.tflib.init_tf()
tf_config = {'rnd.np_random_seed': 1000}


def downscale(images):
    if images.shape[2] > 256:
        factor = images.shape[2] // 256
        images = torch.reshape(images,
                               [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor,
                                factor])
        images = torch.mean(images, dim=(3, 5))
    images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)
    return images


class FID:
    def __init__(self, cfg, num_images, minibatch_size):
        self.num_images = num_images
        self.minibatch_size = minibatch_size
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, encoder, lod):
        distance_measure = pickle.load(open('/data/vgg16_zhang_perceptual.pkl', 'rb'))

        dataset = TFRecordsDataset(self.cfg, logger, rank=0, world_size=1, buffer_size_mb=128,
                                   channels=self.cfg.MODEL.CHANNELS, train=True)

        dataset.reset(lod + 2, self.minibatch_size)
        batches = make_dataloader(self.cfg, logger, dataset, self.minibatch_size, 0,)

        begin = 0
        distance = []
        for idx, x in tqdm(enumerate(batches)):
            end = min(begin + self.minibatch_size, self.num_images)
            if end == self.num_images:
                break
            torch.cuda.set_device(0)
            x = (x / 127.5 - 1.)

            Z = encoder(x, lod, 1)
            Z = Z.repeat(1, mapping.num_layers, 1)

            images = decoder(Z, lod, 1.0, noise=True)

            images = downscale(images)
            images_ref = downscale(torch.tensor(x))

            res = distance_measure.run(images, images_ref, num_gpus=2, assume_frozen=True)
            distance.append(res)
            begin += self.minibatch_size

        logger.info("Result = %f" % (np.asarray(distance).mean()))


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
        ppl = FID(cfg, num_images=10000, minibatch_size=4)
        ppl.evaluate(logger, mapping_fl, decoder, encoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_celeba_sep.yaml',
        world_size=gpu_count, write_log=False)
