# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Perceptual Path Length (PPL)."""

import dnnlib.tflib
import pickle
from net import *
from model import Model
from launcher import run
from dataloader import *
import scipy.linalg

from checkpointer import Checkpointer

from dlutils.pytorch import count_parameters
from dlutils import download
from defaults import get_cfg_defaults
from tqdm import tqdm

import utils

dnnlib.tflib.init_tf()
tf_config = {'rnd.np_random_seed': 1000}

download.from_google_drive('1CIDc9i070KQhHlkr4yIwoJC8xqrwjE0_', directory="metrics")


class FID:
    def __init__(self, cfg, num_images, minibatch_size):
        self.num_images = num_images
        self.minibatch_size = minibatch_size
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, encoder, lod):
        gpu_count = torch.cuda.device_count()
        inception = pickle.load(open('metrics/inception_v3_features.pkl', 'rb'))

        # Sampling loop.
        @utils.cache
        def compute_for_reals(num_images, path, lod):
            dataset = TFRecordsDataset(self.cfg, logger, rank=0, world_size=1, buffer_size_mb=1024, channels=self.cfg.MODEL.CHANNELS, train=True)
            dataset.reset(lod + 2, self.minibatch_size)
            batches = make_dataloader(self.cfg, logger, dataset, self.minibatch_size, 0, numpy=True)

            activations = []
            num_images_processed = 0
            for idx, x in tqdm(enumerate(batches)):
                res = inception.run(x, num_gpus=gpu_count, assume_frozen=True)
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

        mu_real, sigma_real = compute_for_reals(self.num_images, self.cfg.DATASET.PATH, lod)

        dataset = TFRecordsDataset(self.cfg, logger, rank=0, world_size=1, buffer_size_mb=128,
                                   channels=self.cfg.MODEL.CHANNELS, train=True)

        dataset.reset(lod + 2, self.minibatch_size)
        batches = make_dataloader(self.cfg, logger, dataset, self.minibatch_size, 0,)

        activations = []
        num_images_processed = 0
        for idx, x in tqdm(enumerate(batches)):
            torch.cuda.set_device(0)
            x = (x / 127.5 - 1.)

            Z = encoder(x, lod, 1)
            Z = Z.repeat(1, mapping.num_layers, 1)

            images = decoder(Z, lod, 1.0, noise=True)

            images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)

            res = inception.run(images, num_gpus=gpu_count, assume_frozen=True)

            activations.append(res)
            num_images_processed += x.shape[0]
            if num_images_processed > self.num_images:
                break

        activations = np.concatenate(activations)
        print(activations.shape)
        print(self.num_images)

        assert activations.shape[0] >= self.num_images
        activations = activations[:self.num_images]
        assert activations.shape[0] == self.num_images

        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        # Calculate FID.
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)

        logger.info("Result = %f" % (np.real(dist)))


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=None,
        truncation_cutoff=None,
        style_mixing_prob=None,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f
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
    last_epoch = list(extra_checkpoint_data['auxiliary']['scheduler'].values())[0]['last_epoch']
    logger.info("Model trained for %d epochs" % last_epoch)

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Evaluating FID metric")

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

    with torch.no_grad():
        ppl = FID(cfg, num_images=50000, minibatch_size=16 * torch.cuda.device_count())
        ppl.evaluate(logger, mapping_fl, decoder, encoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-fid-reconstruction', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log="metrics/fid_score-reconstruction.txt")
