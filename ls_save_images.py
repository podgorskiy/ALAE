# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Linear Separability (LS)."""
import tensorflow as tf
import torch
import dnnlib
import dnnlib.tflib
import dnnlib.tflib as tflib
import pickle
from dataloader import *
import scipy.linalg

from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR

from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from model_z_gan import Model
import argparse
import logging
import sys
import lreq
from skimage.transform import resize
from tqdm import tqdm

from launcher import run
from PIL import Image
from matplotlib import pyplot as plt
import utils
from net import *

from collections import defaultdict
import numpy as np
import sklearn.svm
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

dnnlib.tflib.init_tf()
tf_config     = {'rnd.np_random_seed': 1000}


class LS:
    def __init__(self, cfg, num_samples, num_keep, attrib_indices, minibatch_gpu):
        assert num_keep <= num_samples
        self.num_samples = num_samples
        self.num_keep = num_keep
        self.attrib_indices = attrib_indices
        self.minibatch_size = minibatch_gpu
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, lod):
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        tfr_writer = tf.python_io.TFRecordWriter("generated_data.000", tfr_opt)
        # Sampling loop.

        rnd = np.random.RandomState(5)

        for _ in tqdm(range(0, self.num_samples, self.minibatch_size)):
            # Generate images.
            torch.cuda.set_device(0)
            latents = rnd.randn(self.minibatch_size, self.cfg.MODEL.LATENT_SPACE_SIZE)
            lat = torch.tensor(latents).float().cuda()

            dlat = mapping(lat)
            images = decoder(dlat, lod, 1.0, True)

            # Downsample to 256x256. The attribute classifiers were built for 256x256.
            if images.shape[2] > 256:
                factor = images.shape[2] // 256
                images = torch.reshape(images,
                                    [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor,
                                     factor])
                images = torch.mean(images, dim=(3, 5))
            images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)

            for i, img in enumerate(images):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                    'lat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lat[i].cpu().numpy().tostring()])),
                    'dlat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dlat[i, 0].cpu().numpy().tostring()]))}))
                tfr_writer.write(ex.SerializeToString())


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

    model.cuda()
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

    logger.info("Evaluating LS metric")

    decoder = nn.DataParallel(decoder)
    mapping_fl = nn.DataParallel(mapping_fl)

    with torch.no_grad():
        ppl = LS(cfg, num_samples=50000, num_keep=10000, attrib_indices=range(40), minibatch_gpu=8)
        ppl.evaluate(logger, mapping_fl, decoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_ffhq_z.yaml',
        world_size=gpu_count, write_log=False)
