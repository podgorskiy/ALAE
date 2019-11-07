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
from model_ae_minist import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
import gc

from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR


def gpu_nnc_predict(trX, trY, teX, batch_size=4096):
    metric_fn = F.pairwise_distance
    idxs = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        mb_idxs = []
        for j in range(0, len(trX), batch_size):
            v1 = teX[i:i+batch_size][:, None, ...]
            v2 = trX[j:j+batch_size][None, :, ...]
            dist = (torch.sum((v1 - v2)**2, dim=2) ** 0.5).detach().cpu().numpy()

            mb_dists.append(np.min(dist, axis=1))
            mb_idxs.append(j + np.argmin(dist, axis=1))
        mb_idxs = np.stack(mb_idxs)
        mb_dists = np.stack(mb_dists)
        i = mb_idxs[np.argmin(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        idxs.append(i)
    idxs = np.concatenate(idxs, axis=0)
    nearest = np.asarray(trY)[idxs]
    return nearest


def eval(cfg, logger, local_rank, world_size, distributed):
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
        encoder=cfg.MODEL.ENCODER,
        mapping_to_latent=cfg.MODEL.MAPPING_TO_LATENT,
        mapping_from_latent=cfg.MODEL.MAPPING_FROM_LATENT
    )
    model.cuda(local_rank)
    model.train()
    model.requires_grad_(False)

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

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_tl,
        'mapping_fl': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    tracker = LossTracker(cfg.OUTPUT_DIR)

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                logger=logger,
                                save=False)

    checkpointer.load()

    layer_to_resolution = decoder.layer_to_resolution

    dataset_train = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=True, needs_labels=True)
    dataset_test = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=False, needs_labels=True)

    model.eval()

    batch_size = cfg.TRAIN.LOD_2_BATCH_1GPU[len(cfg.TRAIN.LOD_2_BATCH_1GPU) - 1]

    dataset_train.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, batch_size)
    dataset_test.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, batch_size)

    batches_train = make_dataloader_y(cfg, logger, dataset_train, batch_size, 0)
    batches_test = make_dataloader_y(cfg, logger, dataset_test, batch_size, 0)

    gc.collect()

    # @utils.cache
    def compute_train():
        train_X = []
        train_Y = []

        for x_orig, y in tqdm(batches_train):
            with torch.no_grad():
                x = (x_orig / 127.5)

                Z = encoder(x, cfg.DATASET.MAX_RESOLUTION_LEVEL, 1)
                train_X += torch.split(Z, 1)
                train_Y += list(y)

        train_X = torch.cat(train_X)
        return train_X, train_Y

    # @utils.cache
    def compute_test():
        test_X = []
        test_Y = []

        for x_orig, y in tqdm(batches_test):
            with torch.no_grad():
                x = (x_orig / 127.5)

                Z = encoder(x, cfg.DATASET.MAX_RESOLUTION_LEVEL, 1)
                test_X += torch.split(Z, 1)
                test_Y += list(y)
        test_X = torch.cat(test_X)
        return test_X, test_Y

    train_X, train_Y = compute_train()
    test_X, test_Y = compute_test()

    m = train_X.mean(dim=0)
    s = (train_X - m).std(dim=0)

    print(train_X.mean(dim=0))
    print(train_X.std(dim=0))

    print(test_X.mean(dim=0))
    print(test_X.std(dim=0))

    train_Y = np.asarray(train_Y)
    test_Y = np.asarray(test_Y)

    d = (train_Y == 0).nonzero()

    #train_X = (train_X - m) / s
    #test_X = (test_X - m) / s

    d = train_X[d]

    print(d.mean(dim=0))
    print(d.std(dim=0))

    prediction = gpu_nnc_predict(train_X, train_Y, test_X)

    acc = metrics.accuracy_score(test_Y, prediction)

    print(acc)

    s = svm.LinearSVC(max_iter=5000, C=0.1)

    s.fit(train_X.cpu(), train_Y)
    prediction = s.predict(test_X.cpu())

    acc = metrics.accuracy_score(test_Y, prediction)

    print(acc)






if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_count = 1
    run(eval, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_z.yaml',
        world_size=gpu_count)
