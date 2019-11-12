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
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
import utils
from tracker import LossTracker
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
from tqdm import tqdm
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
import math
from model_z_gan import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER
    )
    model.cuda(local_rank)
    model.train()

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

        decoder = model.module.decoder
        encoder = model.module.encoder
        mapping_tl = model.module.mapping_tl
        mapping_fl = model.module.mapping_fl
        dlatent_avg = model.module.dlatent_avg
    else:
        decoder = model.decoder
        encoder = model.encoder
        mapping_tl = model.mapping_tl
        mapping_fl = model.mapping_fl
        dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    decoder_optimizer = LREQAdam([
        {'params': decoder.parameters()},
        {'params': mapping_fl.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': mapping_tl.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator_s': encoder,
        'generator': decoder,
        'mapping_tl': mapping_tl,
        'mapping_fl': mapping_fl,
        'dlatent_avg': dlatent_avg
    }
    tracker = LossTracker(cfg.OUTPUT_DIR)

    lod = 4
    batch = 128
    per_gpu_batch = batch // world_size

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()#file_name='results_ae/model_tmp.pth')
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution

    dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS)

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(dataset) * world_size)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer])

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_CLR_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
                                                                batch,
                                                                per_gpu_batch,
                                                                lod2batch.lod,
                                                                2 ** (lod + 2),
                                                                2 ** (lod + 2),
                                                                1,
                                                                len(dataset) * world_size))

        dataset.reset(lod + 2, per_gpu_batch)
        batches = make_dataloader(cfg, logger, dataset, per_gpu_batch, local_rank)

        scheduler.set_batch_size(batch, lod)

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        i = 0
        with torch.autograd.profiler.profile(use_cuda=True, enabled=False) as prof:
            for x_orig in tqdm(batches):
                i +=1
                with torch.no_grad():
                    if x_orig.shape[0] != per_gpu_batch:
                        continue
                    if need_permute:
                        x_orig = x_orig.permute(0, 3, 1, 2)
                    x_orig = (x_orig / 127.5 - 1.)

                    needed_resolution = layer_to_resolution[lod2batch.lod]
                    x = x_orig

                x.requires_grad = True

                encoder_optimizer.zero_grad()
                Z = encoder(x, lod, 1)
                Z_ = mapping_tl(Z)
                loss_d.backward()
                encoder_optimizer.step()

                # if local_rank == 0:
                #     betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                #     model_s.lerp(model, betta)

                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time

                lod_for_saving_model = lod2batch.lod
                lod2batch.step()
                if local_rank == 0:
                    if lod2batch.is_time_to_save():
                        checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
                    if lod2batch.is_time_to_report():
                        save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer, decoder_optimizer)

        scheduler.step()

        if local_rank == 0:
            checkpointer.save("model_tmp_lod%d" % lod_for_saving_model)
            save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer, decoder_optimizer)

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final").wait()


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_z.yaml',
        world_size=gpu_count)
