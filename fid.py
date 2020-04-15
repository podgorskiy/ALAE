# Copyright 2019-2020 Stanislav Pidhorskyi
#
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
import dnnlib.tflib
import pickle
from net import *
from model_z_gan import Model
from launcher import run
from dataloader import *
import scipy.linalg

from checkpointer import Checkpointer

from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from skimage.transform import resize
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt
import utils

dnnlib.tflib.init_tf()
tf_config = {'rnd.np_random_seed': 1000}


class FID:
    def __init__(self, cfg, num_images, minibatch_size):
        self.num_images = num_images
        self.minibatch_size = minibatch_size
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, model, lod):
        # inception = misc.load_pkl('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn') # inception_v3_features.pkl
        inception = pickle.load(open('/data/inception_v3_features.pkl', 'rb'))

        # Sampling loop.
        @utils.cache
        def compute_for_reals(num_images, path, lod):
            dataset = TFRecordsDataset(self.cfg, logger, rank=0, world_size=1, buffer_size_mb=1024, channels=self.cfg.MODEL.CHANNELS, train=True)
            dataset.reset(lod + 2, self.minibatch_size)
            batches = make_dataloader(self.cfg, logger, dataset, self.minibatch_size, 0, numpy=True)

            activations = []
            num_images_processed = 0
            for idx, x in tqdm(enumerate(batches)):
                res = inception.run(x, num_gpus=2, assume_frozen=True)
                activations.append(res)
                num_images_processed += x.shape[0]
                if num_images_processed > num_images:
                    break

            activations = np.concatenate(activations)
            print(activations.shape)
            print(num_images)

            assert activations.shape[0] >= num_images
            activations = activations[:num_images]
            assert activations.shape[0] == num_images

            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            return mu_real, sigma_real

        mu_real, sigma_real = compute_for_reals(25000, self.cfg.DATASET.PATH, lod)

        activations = []
        num_images_processed = 0
        for _ in tqdm(range(0, self.num_images, self.minibatch_size)):
            torch.cuda.set_device(0)
            lat = torch.randn([self.minibatch_size, self.cfg.MODEL.LATENT_SPACE_SIZE])
            dlat = mapping(lat)
            images = decoder(dlat, lod, 1.0, noise=True)
            # images = model.generate(lod, 1, count=self.minibatch_size, no_truncation=True)

            images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)
            #
            # print(images.shape)
            # plt.imshow(images[0].transpose(1, 2, 0), interpolation='nearest')
            # plt.show()

            res = inception.run(images, num_gpus=2, assume_frozen=True)

            activations.append(res)
            if num_images_processed > self.num_images:
                break

        activations = np.concatenate(activations)
        print(activations.shape)
        print(self.num_images)

        assert activations.shape[0] >= self.num_images
        activations = activations[:self.num_images]
        assert activations.shape[0] == self.num_images

        # print("Creating dataset")
        # dataset = TFRecordsDataset(self.cfg, logger, rank=0, world_size=1, buffer_size_mb=1024,
        #                            channels=self.cfg.MODEL.CHANNELS, train=False)
        #
        # dataset.reset(lod + 2, self.minibatch_size)
        # batches = make_dataloader(self.cfg, logger, dataset, self.minibatch_size, 0, numpy=True)
        #
        # activations = []
        # begin = 0
        # for idx, x in tqdm(enumerate(batches)):
        #     torch.cuda.set_device(0)
        #     begin += self.minibatch_size
        #     res = inception.run(x, num_gpus=2, assume_frozen=True)
        #     activations.append(res)
        # activations = np.concatenate(activations)

        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)

        logger.info("Result = %f" % (np.real(dist)))


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
#       truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
#       truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    # mapping_tl = model.mapping_tl
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
        #'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg_s': dlatent_avg
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
        ppl = FID(cfg, num_images=50000, minibatch_size=16)
        ppl.evaluate(logger, mapping_fl, decoder, model, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
